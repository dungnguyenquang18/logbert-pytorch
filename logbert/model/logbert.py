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
    """Content-based pooling trên trục L không chuẩn hoá (bỏ softmax).

    s_t = w · Dropout(h_t) + b
    pooled = Σ_t (s_t · attn_mask)_t · h_t
    """

    def __init__(self, hidden, dropout=0.2):
        super().__init__()
        self.feat_dropout = nn.Dropout(dropout)         # Áp dụng lên biểu diễn đặc trưng
        self.score = nn.Linear(hidden, 1, bias=True)    # weight: [1, hidden], bias: [1]

    def forward(self, x, attn_mask):                    # x: [B, L, H]; attn_mask: [B, L]
        # Regularize đặc trưng trước khi chiếu thành scalar score
        x_dropped = self.feat_dropout(x)
        scores = self.score(x_dropped).squeeze(-1)      # [B, L]
        
        # Triệt tiêu pad/SOS về 0
        alpha = scores * attn_mask                      # [B, L]
        pooled = (alpha.unsqueeze(-1) * x).sum(dim=1)   # [B, H]
        
        return pooled, alpha


class CausalLogModel(nn.Module):
    """Next-log prediction head."""
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class LogBertClassifier(nn.Module):
    """Top-level model for sequence classification."""

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
        attn_mask[:, 0] = 0                                       # drop SOS position
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

        return {
            "logits": logits,
            "loss": loss,
            "loss_cls": loss_cls,
            "loss_causal": loss_causal,
            "attn_weights": alpha
        }