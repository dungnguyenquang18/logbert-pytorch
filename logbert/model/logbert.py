import torch
import torch.nn as nn

from ..config import ModelConfig
from .embedding import BERTEmbedding
from .transformer import TransformerBlock


class BERT(nn.Module):
    """Backbone: embedding + N transformer blocks. Port of modelling/logbert_model.py:BERT."""
    def __init__(self, vocab_size, max_len=20480, hidden=256, n_layers=4, attn_heads=8,
                 dropout=0.1, is_logkey=True, is_time=False, is_device=True,
                 num_devices=None, causal=False):
        super().__init__()
        self.hidden = hidden
        self.feed_forward_hidden = hidden * 2
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, max_len=max_len,
                                       is_logkey=is_logkey, is_time=is_time,
                                       is_device=is_device, num_devices=num_devices)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout, causal)
             for _ in range(n_layers)])

    def forward(self, x, segment_info=None, time_info=None, device_info=None):
        key_padding = (x > 0)
        mask = key_padding[:, None, None, :]
        x = self.embedding(x, segment_info, time_info, device_info)
        for block in self.transformer_blocks:
            x = block(x, mask)
        return x


class AttentionPooling(nn.Module):
    """Content-based pooling trên trục L. Thay {Interpolate + linear1}.

    KHÔNG chuẩn hoá (bỏ softmax): pooled = Σ_t s_t·h_t, với s_t = w·h_t + b.
    Ba hệ quả có chủ đích:

    1. Ngữ nghĩa TỔNG chứ không phải trung bình — log volume đi thẳng vào
       ‖pooled‖. Với NOC, volume spike là chỉ báo sự cố thật, không phải nhiễu
       cần khử. (Softmax ép Σα=1 nên chỉ đọc được *thành phần*, bỏ mất *khối lượng*.)
    2. s_t CÓ DẤU ⇒ cặp (score, cls_head) có gauge tự do: lật dấu cả hai cho ra
       đúng cùng một hàm số. Nên dấu của tương quan giữa s_t và thống kê ngoài
       KHÔNG khả định danh — chỉ |ρ| là đại lượng xác định, đúng như
       ClassifierHead {linear1, linear2} của DeviceIncidents. Đại lượng RCA bất
       biến gauge là c_t = s_t · g(h_t), g(h_t) = (cls_head.weight[1] -
       cls_head.weight[0])·h_t, thoả Σ_t c_t + const = logit[1] - logit[0].
    3. b (bias) điều khiển tỉ lệ trộn sum-pooling thuần: pooled = Σ_t (w·h_t)·h_t
       + b·Σ_t h_t. Dưới softmax thì bias vô hướng là no-op tuyệt đối (gradient
       đúng bằng 0); bỏ softmax rồi nó mới có tác dụng.

    attn_mask nhân vào s_t là BẮT BUỘC, không phải phòng thủ thừa: LogCollator
    pad tới chuỗi dài nhất *trong batch*, nên nếu pad lọt vào tổng thì cùng một
    chuỗi sẽ cho pooled khác nhau tuỳ thành phần batch.
    """

    def __init__(self, hidden, dropout=0.2):
        super().__init__()
        self.score = nn.Linear(hidden, 1, bias=True)   # weight: [1, hidden], bias: [1]
        self.attn_dropout = nn.Dropout(dropout)   # regularize s_t; no-op in eval

    def forward(self, x, attn_mask):           # x:[B,L,H]; attn_mask:[B,L] (1=valid, 0=pad/SOS)
        scores = self.score(x).squeeze(-1)                  # [B, L]
        alpha  = scores * attn_mask                         # pad/SOS đóng góp đúng 0
        alpha  = self.attn_dropout(alpha)
        pooled = (alpha.unsqueeze(-1) * x).sum(dim=1)   # [B,L,1] * [B,L,H] → [B,L,H] → sum L → [B,H]
        return pooled, alpha


class CausalLogModel(nn.Module):
    """Next-log prediction head. Combined with causal attention (BERT's
    causal=True), this makes loss_causal a standard autoregressive LM loss,
    not a bidirectional masked-LM loss."""
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class LogBertClassifier(nn.Module):
    """Top-level model. Replaces HF LogBertForSequenceClassification;
    forward returns a plain dict instead of SequenceClassifierOutput."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.bert = BERT(vocab_size=cfg.vocab_size, max_len=cfg.max_seq_len, hidden=cfg.hidden,
                         n_layers=cfg.layers, attn_heads=cfg.attn_heads, dropout=cfg.dropout,
                         is_logkey=True, is_time=False, is_device=cfg.is_device,
                         num_devices=cfg.num_devices, causal=cfg.causal)
        self.pool = AttentionPooling(cfg.hidden, dropout=cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)
        self.cls_head = nn.Linear(cfg.hidden, cfg.num_labels)
        self.causal_lm_head = CausalLogModel(cfg.hidden, cfg.vocab_size) if cfg.use_causal_lm else None
        self.register_buffer("class_weight", None)
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.causal_loss_fn = nn.NLLLoss(ignore_index=0)
        # Same init the HF post_init applied: normal(0, 0.02) on Linear/Embedding, zero bias.
        # NOTE: this intentionally overwrites ClassifierHead.linear1's constant init,
        # exactly as the old post_init did.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def set_class_weight(self, weight):
        if weight is None:
            self.class_weight = None
        else:
            self.register_buffer("class_weight", weight.clone())

    def forward(self, input_ids, device_ids=None, labels=None, causal_labels=None):
        x = self.bert(input_ids, device_info=device_ids)          # [B, L, H]

        attn_mask = (input_ids > 0).float()
        attn_mask[:, 0] = 0                                        # drop SOS position
        pooled, alpha = self.pool(x, attn_mask)
        logits = self.cls_head(self.dropout(pooled))

        loss = loss_cls = loss_causal = None
        if labels is not None:
            if self.training and self.class_weight is not None:
                loss_cls = nn.functional.cross_entropy(logits, labels,
                                                       weight=self.class_weight, reduction="mean")
            else:
                loss_cls = self.loss_fn(logits, labels)
            loss = loss_cls
        if self.causal_lm_head is not None and causal_labels is not None:
            valid = causal_labels != self.causal_loss_fn.ignore_index
            if valid.any():
                loss_causal = self.causal_loss_fn(self.causal_lm_head(x).transpose(1, 2), causal_labels)
            else:
                loss_causal = x.new_zeros(())
            loss = loss_causal if loss is None else loss + self.cfg.alpha_causal_lm * loss_causal

        return {"logits": logits, "loss": loss,
                "loss_cls": loss_cls, "loss_causal": loss_causal, "attn_weights": alpha}

