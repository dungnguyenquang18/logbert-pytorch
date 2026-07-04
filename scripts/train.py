"""Train entrypoint.
Single GPU:  python scripts/train.py --config configs/default.py
Multi GPU:   torchrun --nproc_per_node=2 scripts/train.py --config configs/default.py
"""
import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logbert.data.log_sequences import LogCollator, LogSequenceDataset, load_pkl_dir
from logbert.data.sampler import BucketBatchSampler
from logbert.model import LogBertClassifier
from logbert.training.trainer import Trainer
from logbert.vocab import DeviceVocab, WordVocab

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")


def load_config_module(path: str):
    spec = importlib.util.spec_from_file_location("exp_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ("model", "data", "train"):
        if not hasattr(mod, attr):
            raise AttributeError(f"config {path} must define `{attr}`")
    return mod


def calc_class_weight(labels) -> torch.Tensor:
    if len(labels) == 0:
        return torch.ones(2, dtype=torch.float)
    counts = np.bincount(np.array(labels), minlength=2)
    if counts.sum() == 0:
        return torch.ones(2, dtype=torch.float)
    weights = counts.sum() / (len(counts) * counts.astype(np.float32).clip(min=1))
    return torch.tensor(weights, dtype=torch.float)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config_path: str):
    cfg = load_config_module(config_path)
    model_cfg, data_cfg, train_cfg = cfg.model, cfg.data, cfg.train

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    set_seed(train_cfg.seed)

    vocab = WordVocab.load_vocab(data_cfg.vocab_path)
    device_vocab = DeviceVocab.load_vocab(data_cfg.device_vocab_path)
    model_cfg.vocab_size = len(vocab)
    model_cfg.num_devices = len(device_vocab)
    logger.info(f"vocab={len(vocab)} device_vocab={len(device_vocab)}")

    train_seqs = load_pkl_dir(data_cfg.train_dir)
    train_ds = LogSequenceDataset(train_seqs, vocab=vocab, mask_ratio=data_cfg.mask_ratio)
    collator = LogCollator(vocab=vocab, seq_len=data_cfg.seq_len)

    def make_loader(ds, mode):
        return DataLoader(ds, batch_sampler=BucketBatchSampler(
            ds, batch_size=data_cfg.batch_size, seed=train_cfg.seed, mode=mode),
            collate_fn=collator, num_workers=data_cfg.num_workers,
            pin_memory=torch.cuda.is_available())

    train_dl = make_loader(train_ds, "train")
    eval_dl = None
    if data_cfg.valid_dir and os.path.exists(data_cfg.valid_dir):
        eval_ds = LogSequenceDataset(load_pkl_dir(data_cfg.valid_dir), vocab=vocab)
        eval_dl = make_loader(eval_ds, "eval")
    logger.info(f"train={len(train_ds)} eval={0 if eval_dl is None else len(eval_dl.dataset)}")

    model = LogBertClassifier(model_cfg)
    labels = [t[3] for t in train_seqs if len(t) > 3 and int(t[3]) >= 0]
    cw = calc_class_weight(labels)
    logger.info(f"class weights: {cw.tolist()}")
    model.set_class_weight(cw)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"parameters: {total:,}")

    Trainer(model, train_dl, eval_dl, train_cfg).fit()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.py")
    main(p.parse_args().config)
