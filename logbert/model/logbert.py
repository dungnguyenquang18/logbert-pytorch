import torch
import torch.nn as nn

from ..config import ModelConfig
from .embedding import BERTEmbedding
from .interpolate import Interpolate
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


class LDropout(nn.Module):
    """Drops whole positions of the [B, H, L] classifier input."""
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        B, H, L = x.shape
        mask = (torch.rand(B, 1, L, device=x.device) > self.p).float()
        return x * mask


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


class ClassifierHead(nn.Module):
    """Aggregates the L dimension with a learned linear filter (whose weights are
    the per-position influence scores used for root-cause analysis)."""
    def __init__(self, hidden, max_len_seq, num_classes=2):
        super().__init__()
        self.max_len_seq = max_len_seq - 1
        self.dropout1 = LDropout(0.1)
        self.linear1 = nn.Linear(self.max_len_seq, 1)
        self.dropout2 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(hidden, num_classes)

    def forward(self, x):                 # x: [B, L, H]
        x = x.transpose(1, 2)             # [B, H, L]
        x = self.dropout1(x)
        x = self.linear1(x).squeeze(-1)   # [B, H]
        return self.linear2(self.dropout2(x))


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
        self.interpolate = Interpolate(max_len_seq=cfg.max_seq_len)
        self.classifier = ClassifierHead(hidden=cfg.hidden, max_len_seq=cfg.max_seq_len,
                                         num_classes=cfg.num_labels)
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
        x_norm = self.interpolate.normalize_sequence(x, attn_mask=attn_mask)
        logits = self.classifier(x_norm)

        loss = loss_cls = loss_causal = loss_l1 = None
        if labels is not None:
            if self.training and self.class_weight is not None:
                loss_cls = nn.functional.cross_entropy(logits, labels,
                                                       weight=self.class_weight, reduction="mean")
            else:
                loss_cls = self.loss_fn(logits, labels)
            loss = loss_cls
        if self.cfg.use_l1:
            loss_l1 = torch.norm(self.classifier.linear1.weight, 1)
            loss = loss_l1 * self.cfg.lambda_l1 if loss is None else loss + self.cfg.lambda_l1 * loss_l1
        if self.causal_lm_head is not None and causal_labels is not None:
            valid = causal_labels != self.causal_loss_fn.ignore_index
            if valid.any():
                loss_causal = self.causal_loss_fn(self.causal_lm_head(x).transpose(1, 2), causal_labels)
            else:
                loss_causal = x.new_zeros(())
            loss = loss_causal if loss is None else loss + self.cfg.alpha_causal_lm * loss_causal

        return {"logits": logits, "loss": loss,
                "loss_cls": loss_cls, "loss_causal": loss_causal, "loss_l1": loss_l1}
