"""Default experiment config. Copy this file per experiment and edit."""
from logbert.config import DataConfig, ModelConfig, TrainConfig

_OLD = "/home/dungnq/study/5Y1S/meobtnx/DeviceIncidents/output_dungnq57_v2.7.8"

model = ModelConfig(
    vocab_size=0,          # filled at runtime from vocab.pkl
    hidden=256, layers=1, attn_heads=4,
    max_seq_len=36000, causal=True, use_mlm=True, use_l1=True,
    alpha_mlm=0.1, lambda_l1=1e-4, is_device=False,
)

data = DataConfig(
    train_dir=f"{_OLD}/train/",
    valid_dir=f"{_OLD}/valid/",
    test_dir=f"{_OLD}/test_org/",
    vocab_path=f"{_OLD}/vocab.pkl",
    device_vocab_path=f"{_OLD}/dev_vocab.pkl",
    seq_len=36000, mask_ratio=0.0, batch_size=8,
)

train = TrainConfig(
    output_dir="outputs/default/",
    epochs=20, lr=5e-5, grad_accum_steps=32, bf16=True,
)
