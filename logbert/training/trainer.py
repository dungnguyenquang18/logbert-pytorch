"""Hand-written training loop. Reads top-to-bottom; no callback framework.
Replaces HF Trainer + BucketTrainer + 4 TrainerCallbacks (~800 lines → ~300)."""
import gc
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from ..config import TrainConfig
from .checkpoint import load_checkpoint, rotate_checkpoints, save_checkpoint
from .scheduler import build_scheduler

logger = logging.getLogger(__name__)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))


class Trainer:
    def __init__(self, model, train_loader, eval_loader, cfg: TrainConfig):
        self.cfg = cfg
        self.world_size, self.rank = get_dist_info()
        self.is_main = self.rank == 0

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            self.device = torch.device("cuda", local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.raw_model = model.to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.raw_model,
                             device_ids=[local_rank] if self.device.type == "cuda" else None,
                             find_unused_parameters=False)
        else:
            self.model = self.raw_model

        self.train_loader = train_loader
        self.eval_loader = eval_loader

        lr = cfg.lr
        if cfg.scale_lr_with_world_size and self.world_size > 1:
            lr = cfg.lr * self.world_size * 0.68        # empirical scale from old repo
            logger.info(f"DDP lr scale: {cfg.lr} -> {lr} (world_size={self.world_size})")
        self.optimizer = torch.optim.AdamW(self.raw_model.parameters(), lr=lr,
                                           betas=tuple(cfg.betas),
                                           weight_decay=cfg.weight_decay)

        self.steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps))
        total_steps = self.steps_per_epoch * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.scheduler = build_scheduler(self.optimizer, cfg.scheduler_type,
                                         warmup_steps, total_steps)
        logger.info(f"steps/epoch={self.steps_per_epoch} total={total_steps} warmup={warmup_steps}")

        self.global_step = 0
        self.start_epoch = 0
        self.best_eval = None
        self.bad_evals = 0
        self.should_stop = False
        self.history = []

        self.output_dir = Path(cfg.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.tb = None
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb_logs"))

        self.autocast_enabled = cfg.bf16 and self.device.type == "cuda"

        if cfg.resume_from:
            self._resume(cfg.resume_from)

    # ---------- logging ----------
    def _log(self, entry: dict, step=None):
        if not self.is_main:
            return
        step = self.global_step if step is None else step
        entry = {"step": step,
                 "epoch": round(step / self.steps_per_epoch, 4), **entry}
        self.history.append(entry)
        (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2))
        for k, v in entry.items():
            if k not in ("step", "epoch") and isinstance(v, (int, float)):
                self.tb.add_scalar(k, v, step)
        parts = " ".join(f"{k}={v:.6g}" for k, v in entry.items()
                         if isinstance(v, (int, float)))
        logger.info(f"[{parts}]")

    @staticmethod
    def _component_means(sums: dict, count: int, prefix: str) -> dict:
        out = {}
        for key, total in sums.items():
            name = "loss" if (key == "loss" and prefix == "train") else f"{prefix}_{key}"
            out[name] = total / count
        return out

    # ---------- forward helpers ----------
    def _forward(self, batch):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast_enabled):
            return self.model(input_ids=batch["input_ids"], device_ids=batch["device_ids"],
                              labels=batch["labels"], causal_labels=batch["causal_labels"])

    @staticmethod
    def _accumulate(sums: dict, out: dict):
        for key in ("loss", "loss_cls", "loss_causal", "loss_l1"):
            if out.get(key) is not None:
                sums[key] = sums.get(key, 0.0) + out[key].detach().float().item()

    # ---------- initial loss (sanity check at step 0) ----------
    def _log_initial_loss(self):
        entry = {}
        self.model.train()                       # train-mode loss, but no backward
        with torch.no_grad():
            out = self._forward(next(iter(self.train_loader)))
            for key in ("loss", "loss_cls", "loss_causal", "loss_l1"):
                if out.get(key) is not None:
                    entry[f"initial_train_{key}" if key != "loss" else "initial_train_loss"] = \
                        out[key].detach().float().item()
        if self.eval_loader is not None:
            self.model.eval()
            with torch.no_grad():
                out = self._forward(next(iter(self.eval_loader)))
                for key in ("loss", "loss_cls", "loss_causal", "loss_l1"):
                    if out.get(key) is not None:
                        entry[f"initial_eval_{key}" if key != "loss" else "initial_eval_loss"] = \
                            out[key].detach().float().item()
        self._log(entry, step=0)

    # ---------- eval ----------
    def evaluate(self) -> dict:
        self.model.eval()
        sums, count = {}, 0
        with torch.no_grad():
            for batch in self.eval_loader:
                self._accumulate(sums, self._forward(batch))
                count += 1
        means = {k: v / max(1, count) for k, v in sums.items()}
        if self.world_size > 1:                   # average across ranks
            keys = sorted(means)
            t = torch.tensor([means[k] for k in keys], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            means = dict(zip(keys, t.tolist()))
        metrics = {("eval_loss" if k == "loss" else f"eval_{k}"): v for k, v in means.items()}
        self._log(metrics)
        self.model.train()
        return metrics

    def _eval_and_maybe_stop(self):
        if self.eval_loader is None:
            return
        eval_loss = self.evaluate().get("eval_loss")
        if eval_loss is None:
            return
        if self.best_eval is None or eval_loss < self.best_eval - self.cfg.min_delta:
            logger.info(f"eval improved: {self.best_eval} -> {eval_loss}")
            self.best_eval = eval_loss
            self.bad_evals = 0
        else:
            self.bad_evals += 1
            logger.info(f"no improvement ({self.bad_evals}/{self.cfg.early_stop_patience})")
            if self.bad_evals >= self.cfg.early_stop_patience:
                logger.warning("early stopping")
                self.should_stop = True

    # ---------- checkpointing ----------
    def _save(self, epoch):
        if self.world_size > 1:
            dist.barrier()
        if self.is_main:
            path = self.ckpt_dir / f"ckpt_step_{self.global_step}.pt"
            save_checkpoint(path, self.raw_model, self.optimizer, self.scheduler,
                            epoch=epoch, global_step=self.global_step,
                            best_eval=self.best_eval, bad_evals=self.bad_evals,
                            config=vars(self.cfg))
            rotate_checkpoints(self.ckpt_dir, self.cfg.save_total_limit)
        if self.world_size > 1:
            dist.barrier()

    def _resume(self, path):
        meta = load_checkpoint(path, self.raw_model, self.optimizer, self.scheduler)
        self.start_epoch = meta["epoch"] + 1          # resume at next epoch
        self.global_step = meta["global_step"]
        self.best_eval = meta["best_eval"]
        self.bad_evals = meta["bad_evals"]
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "epoch"):
            sampler.epoch = self.start_epoch          # keep shuffle order aligned
        logger.info(f"resumed from {path}: epoch={self.start_epoch} step={self.global_step}")

    def _save_final(self):
        if not self.is_main:
            return
        # class_weight is training-only (loss never applies it in eval mode) and is not
        # registered on a freshly constructed model, so a fresh load_state_dict would
        # reject it as an unexpected key.
        state_dict = {k: v for k, v in self.raw_model.state_dict().items()
                     if k != "class_weight"}
        torch.save({"model": state_dict,
                    "config": vars(self.raw_model.cfg)},
                   self.output_dir / "model_final.pt")
        logger.info(f"saved final model to {self.output_dir / 'model_final.pt'}")

    # ---------- the loop ----------
    def fit(self):
        cfg = self.cfg
        if cfg.log_initial_loss and self.global_step == 0 and self.is_main:
            self._log_initial_loss()

        self.model.train()
        sums, n_micro = {}, 0
        for epoch in range(self.start_epoch, cfg.epochs):
            self.optimizer.zero_grad(set_to_none=True)
            num_micro = len(self.train_loader)
            for i, batch in enumerate(self.train_loader):
                out = self._forward(batch)
                (out["loss"] / cfg.grad_accum_steps).backward()
                self._accumulate(sums, out)
                n_micro += 1

                if (i + 1) % cfg.grad_accum_steps == 0 or (i + 1) == num_micro:
                    torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    if cfg.log_every > 0 and self.global_step % cfg.log_every == 0 and n_micro:
                        entry = self._component_means(sums, n_micro, "train")
                        entry["learning_rate"] = self.scheduler.get_last_lr()[0]
                        self._log(entry)
                        sums, n_micro = {}, 0
                    if cfg.eval_every > 0 and self.global_step % cfg.eval_every == 0:
                        self._eval_and_maybe_stop()
                    if cfg.save_every > 0 and self.global_step % cfg.save_every == 0:
                        self._save(epoch)
                    if self.should_stop:
                        break

            if n_micro:                               # flush remaining train stats
                entry = self._component_means(sums, n_micro, "train")
                entry["learning_rate"] = self.scheduler.get_last_lr()[0]
                self._log(entry)
                sums, n_micro = {}, 0
            if cfg.eval_every == 0 and not self.should_stop:
                self._eval_and_maybe_stop()
            if cfg.save_every == 0:
                self._save(epoch)
            if cfg.cuda_clear_each_epoch and torch.cuda.is_available():
                torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
            if self.should_stop:
                break

        self._save_final()
        if self.tb is not None:
            self.tb.close()
