import torch

from logbert.config import ModelConfig
from logbert.model import LogBertClassifier


def small_cfg(**over):
    base = dict(vocab_size=30, hidden=32, layers=1, attn_heads=2,
                max_seq_len=64, causal=True, use_causal_lm=True, use_l1=True,
                is_device=False, num_devices=6)
    base.update(over)
    return ModelConfig(**base)


def make_batch(B=4, L=20, vocab=30):
    torch.manual_seed(0)
    input_ids = torch.randint(5, vocab, (B, L))
    input_ids[:, 0] = 3  # SOS
    return {
        "input_ids": input_ids,
        "device_ids": torch.zeros(B, L, dtype=torch.long),
        "labels": torch.tensor([0, 1, 0, 1]),
        "causal_labels": torch.randint(5, vocab, (B, L)),
    }


def test_forward_returns_all_components():
    m = LogBertClassifier(small_cfg())
    out = m(**make_batch())
    assert out["logits"].shape == (4, 2)
    for k in ("loss", "loss_cls", "loss_causal", "loss_l1"):
        assert out[k] is not None and out[k].dim() == 0
    # composition: loss = cls + lambda_l1*l1 + alpha_causal_lm*causal
    expected = out["loss_cls"] + 1e-4 * out["loss_l1"] + 0.1 * out["loss_causal"]
    assert torch.allclose(out["loss"], expected, atol=1e-6)


def test_heads_disabled():
    m = LogBertClassifier(small_cfg(use_causal_lm=False, use_l1=False))
    out = m(**make_batch())
    assert out["loss_causal"] is None and out["loss_l1"] is None
    assert torch.allclose(out["loss"], out["loss_cls"])


def test_no_labels_predict_mode():
    m = LogBertClassifier(small_cfg(use_l1=False))
    b = make_batch(); b["labels"] = None; b["causal_labels"] = None
    out = m(**b)
    assert out["logits"].shape == (4, 2) and out["loss"] is None


def test_class_weight_train_only():
    torch.manual_seed(0)
    m = LogBertClassifier(small_cfg(use_causal_lm=False, use_l1=False))
    m.set_class_weight(torch.tensor([1.0, 10.0]))
    b = make_batch()
    m.train(); loss_train = m(**b)["loss_cls"]
    m.eval()
    with torch.no_grad():
        loss_eval = m(**b)["loss_cls"]
    # weighted vs unweighted must differ
    assert abs(loss_train.item() - loss_eval.item()) > 1e-6


def test_backward_flows():
    m = LogBertClassifier(small_cfg())
    out = m(**make_batch())
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
