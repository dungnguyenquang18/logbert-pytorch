"""Single-file .pt checkpoints: weights + optimizer + scheduler + counters + RNG."""
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path, model, optimizer, scheduler, *, epoch, global_step,
                    best_eval, bad_evals, config=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_eval": best_eval,
        "bad_evals": bad_evals,
        "config": config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)                      # atomic-ish: no torn file on crash


def load_checkpoint(path, model, optimizer=None, scheduler=None, restore_rng=True) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng and state.get("rng"):
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return {k: state[k] for k in ("epoch", "global_step", "best_eval", "bad_evals", "config")}


def rotate_checkpoints(ckpt_dir, limit: int):
    files = sorted(Path(ckpt_dir).glob("ckpt_step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    for old in files[:-limit] if limit > 0 else []:
        old.unlink()
