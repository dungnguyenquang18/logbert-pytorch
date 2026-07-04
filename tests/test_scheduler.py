import math

import pytest
import torch

from logbert.training.scheduler import build_scheduler


def make(kind, warmup=10, total=100, lr=1.0):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    return opt, build_scheduler(opt, kind, warmup, total)


def lr_at(opt, sched, step):
    for _ in range(step):
        opt.step(); sched.step()
    return opt.param_groups[0]["lr"]


def test_linear_warmup_then_decay():
    opt, sched = make("linear")
    assert lr_at(opt, sched, 5) == pytest.approx(0.5)      # halfway up warmup
    opt, sched = make("linear")
    assert lr_at(opt, sched, 10) == pytest.approx(1.0)     # peak
    opt, sched = make("linear")
    assert lr_at(opt, sched, 100) == pytest.approx(0.0)    # decayed to 0


def test_cosine():
    opt, sched = make("cosine")
    assert lr_at(opt, sched, 55) == pytest.approx(0.5, abs=1e-6)  # cos midpoint
    opt, sched = make("cosine")
    assert lr_at(opt, sched, 100) == pytest.approx(0.0, abs=1e-6)


def test_constant_with_warmup():
    opt, sched = make("constant_with_warmup")
    assert lr_at(opt, sched, 50) == pytest.approx(1.0)


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        make("polynomial")
