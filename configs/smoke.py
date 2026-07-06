"""Smoke experiment: tiny model on generated mock data — verifies the flow only.
Run order (from repo root):
    python scripts/make_mock_data.py --out mock_data/
    python scripts/train.py --config configs/smoke.py
    python scripts/predict.py --config configs/smoke.py --model outputs/smoke/model_final.pt
    python scripts/plot_loss.py --history outputs/smoke/history.json --output-dir outputs/smoke/plots/
    python scripts/visualize_weights.py --model outputs/smoke/model_final.pt --output-dir outputs/smoke/plots/
"""
from logbert.config import DataConfig, ModelConfig, TrainConfig

model = ModelConfig(
    vocab_size=0,              # filled at runtime from vocab.pkl
    hidden=32, layers=1, attn_heads=2,
    max_seq_len=64, causal=True, use_causal_lm=True,
    alpha_causal_lm=0.1, is_device=False,
)

data = DataConfig(
    train_dir="mock_data/train/",
    valid_dir="mock_data/valid/",
    test_dir="mock_data/test_org/",
    vocab_path="mock_data/vocab.pkl",
    device_vocab_path="mock_data/dev_vocab.pkl",
    seq_len=64, mask_ratio=0.0, batch_size=16, num_workers=0,
)

train = TrainConfig(
    output_dir="outputs/smoke/",
    epochs=15, lr=3e-3, grad_accum_steps=1, bf16=True,
    log_every=10, eval_every=0, save_every=0, save_total_limit=2,
    early_stop_patience=10, min_delta=0.0,
)
