import torch

from logbert.config import ModelConfig
from logbert.model import LogBertClassifier


def small_cfg(**over):
    base = dict(vocab_size=30, hidden=32, layers=1, attn_heads=2,
                max_seq_len=64, causal=True, use_causal_lm=True,
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
    m.eval()  # attn_dropout must be off for the exact pooling identity below
    batch = make_batch()
    out = m(**batch)
    assert out["logits"].shape == (4, 2)
    for k in ("loss", "loss_cls", "loss_causal"):
        assert out[k] is not None and out[k].dim() == 0
    # composition: loss = cls + alpha_causal_lm*causal
    expected = out["loss_cls"] + 0.1 * out["loss_causal"]
    assert torch.allclose(out["loss"], expected, atol=1e-6)

    # attn_weights validation — s_t thô, KHÔNG chuẩn hoá (không còn Σα=1)
    alpha = out["attn_weights"]
    assert alpha.shape == (4, 20)
    # weights on SOS and pad (<=0 in input_ids) should be 0
    mask = (batch["input_ids"] > 0).float()
    mask[:, 0] = 0
    assert torch.allclose(alpha * (1 - mask), torch.zeros_like(alpha), atol=1e-6)


def test_heads_disabled():
    m = LogBertClassifier(small_cfg(use_causal_lm=False))
    out = m(**make_batch())
    assert out["loss_causal"] is None
    assert torch.allclose(out["loss"], out["loss_cls"])


def test_no_labels_predict_mode():
    m = LogBertClassifier(small_cfg())
    b = make_batch(); b["labels"] = None; b["causal_labels"] = None
    out = m(**b)
    assert out["logits"].shape == (4, 2) and out["loss"] is None


def test_class_weight_train_only():
    torch.manual_seed(0)
    m = LogBertClassifier(small_cfg(use_causal_lm=False))
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


def test_attention_pooling_masks_pad_and_sos():
    """pad/SOS phải đóng góp đúng 0. Không phải phòng thủ thừa: LogCollator pad
    tới chuỗi dài nhất TRONG BATCH, nên pad lọt vào tổng ⇒ cùng một chuỗi cho
    pooled khác nhau tuỳ thành phần batch."""
    from logbert.model import AttentionPooling
    torch.manual_seed(42)
    B, L, H = 2, 5, 8
    x = torch.randn(B, L, H)
    input_ids = torch.tensor([
        [3, 5, 6, 0, 0],
        [3, 7, 8, 9, 0],
    ])
    attn_mask = (input_ids > 0).float()
    attn_mask[:, 0] = 0  # drop SOS

    pool = AttentionPooling(H)
    pool.eval()
    pooled, alpha = pool(x, attn_mask)

    assert torch.allclose(alpha * (1 - attn_mask), torch.zeros_like(alpha), atol=1e-6)
    # pooled = Σ_t s_t·h_t — đẳng thức chính xác, nền tảng của phân rã RCA
    assert torch.allclose(pooled, torch.einsum("bl,blh->bh", alpha, x), atol=1e-6)
    # và KHÔNG còn Σα=1
    assert not torch.allclose(alpha.sum(dim=1), torch.ones(B), atol=1e-3)


def test_attention_pooling_bias_is_live():
    """Dưới softmax, bias vô hướng là no-op tuyệt đối (gradient đúng bằng 0).
    Bỏ softmax rồi nó mới có tác dụng: pooled += b·Σ_t h_t."""
    from logbert.model import AttentionPooling
    torch.manual_seed(0)
    B, L, H = 2, 6, 8
    x = torch.randn(B, L, H)
    attn_mask = torch.ones(B, L)

    pool = AttentionPooling(H)
    pool.eval()
    pooled_a, _ = pool(x, attn_mask)
    with torch.no_grad():
        pool.score.bias += 1.0
    pooled_b, _ = pool(x, attn_mask)

    assert torch.allclose(pooled_b - pooled_a, x.sum(dim=1), atol=1e-5)


def test_sign_gauge_symmetry_of_score_and_cls_head():
    """Lật dấu (score, cls_head.weight) cho ra ĐÚNG cùng một hàm số ⇒ dấu của
    s_t không khả định danh. Đây là cơ sở toán học để báo cáo |ρ| thay vì ρ,
    giống ClassifierHead {linear1, linear2} của ART v1. Softmax phá đối xứng
    này (α ≥ 0 không lật được) — nên v1 báo cáo |ρ| là hợp lệ."""
    m = LogBertClassifier(small_cfg(use_causal_lm=False))
    m.eval()
    batch = make_batch()
    logits_before = m(input_ids=batch["input_ids"], device_ids=batch["device_ids"])["logits"]

    with torch.no_grad():
        m.pool.score.weight.neg_()
        m.pool.score.bias.neg_()
        m.cls_head.weight.neg_()   # cls_head.bias giữ nguyên

    logits_after = m(input_ids=batch["input_ids"], device_ids=batch["device_ids"])["logits"]
    assert torch.allclose(logits_before, logits_after, atol=1e-5)


def test_attention_pooling_dropout_active_in_train_mode():
    from logbert.model import AttentionPooling
    torch.manual_seed(0)
    B, L, H = 4, 10, 8
    x = torch.randn(B, L, H)
    attn_mask = torch.ones(B, L)

    pool = AttentionPooling(H, dropout=0.5)
    pool.train()
    _, alpha_train = pool(x, attn_mask)
    assert (alpha_train == 0).any()   # dropout zeroes some scores at train time

    pool.eval()
    _, alpha_eval = pool(x, attn_mask)
    assert not (alpha_eval == 0).any()   # no-op at eval

