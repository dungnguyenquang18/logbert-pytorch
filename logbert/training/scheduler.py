"""Warmup LR schedules — the two shapes the old HF configs actually used."""
import math

from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(optimizer, kind: str, warmup_steps: int, total_steps: int) -> LambdaLR:
    def warmup(step):
        return step / max(1, warmup_steps)

    def linear(step):
        if step < warmup_steps:
            return warmup(step)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    def cosine(step):
        if step < warmup_steps:
            return warmup(step)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def constant_with_warmup(step):
        return warmup(step) if step < warmup_steps else 1.0

    fns = {"linear": linear, "cosine": cosine, "constant_with_warmup": constant_with_warmup}
    return LambdaLR(optimizer, fns[kind])
