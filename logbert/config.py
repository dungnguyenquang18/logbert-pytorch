"""All experiment configuration. One experiment = one file in configs/
instantiating these three dataclasses (see configs/default.py)."""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    vocab_size: int                    # filled at runtime from len(vocab)
    hidden: int = 256
    layers: int = 1
    attn_heads: int = 4
    max_seq_len: int = 36000           # max log sequence length for PositionalEmbedding buffer
    dropout: float = 0.1
    causal: bool = True
    use_causal_lm: bool = True         # next-log (causal LM) prediction head
    alpha_causal_lm: float = 0.1       # weight of causal LM loss (was hardcoded 0.1 in old modeling.py:176)
    num_labels: int = 2
    is_device: bool = False
    num_devices: int = 1               # filled at runtime from len(device_vocab)


@dataclass
class DataConfig:
    train_dir: str
    vocab_path: str
    device_vocab_path: str
    valid_dir: Optional[str] = None
    test_dir: Optional[str] = None
    seq_len: int = 36000
    mask_ratio: float = 0.0
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class TrainConfig:
    output_dir: str
    epochs: int = 20
    lr: float = 5e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    scheduler_type: str = "linear"     # linear | cosine | constant_with_warmup
    grad_accum_steps: int = 32
    max_grad_norm: float = 1.0         # HF Trainer default — must keep
    bf16: bool = True
    seed: int = 42
    log_every: int = 50                # optimizer steps; 0 = end of epoch
    eval_every: int = 0                # optimizer steps; 0 = end of epoch
    save_every: int = 0                # optimizer steps; 0 = end of epoch
    save_total_limit: int = 50
    early_stop_patience: int = 4
    min_delta: float = 0.002
    resume_from: Optional[str] = None  # path to ckpt_step_N.pt
    scale_lr_with_world_size: bool = True   # lr × world_size × 0.68 under DDP
    cuda_clear_each_epoch: bool = True
    log_initial_loss: bool = True
