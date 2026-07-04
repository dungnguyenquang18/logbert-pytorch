import random

import numpy as np
import torch

from logbert.training.checkpoint import load_checkpoint, rotate_checkpoints, save_checkpoint


def make_model_opt():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 2)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    s = torch.optim.lr_scheduler.LambdaLR(o, lambda step: 1.0)
    return m, o, s


def test_roundtrip_restores_everything(tmp_path):
    m, o, s = make_model_opt()
    # take a step so optimizer has state
    m(torch.randn(3, 4)).sum().backward(); o.step(); s.step()
    path = tmp_path / "ckpt_step_10.pt"
    save_checkpoint(path, m, o, s, epoch=2, global_step=10, best_eval=0.5, bad_evals=1)

    m2, o2, s2 = make_model_opt()
    meta = load_checkpoint(path, m2, o2, s2)
    assert meta["epoch"] == 2 and meta["global_step"] == 10
    assert meta["best_eval"] == 0.5 and meta["bad_evals"] == 1
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    assert str(o2.state_dict()) == str(o.state_dict())


def test_rng_restored(tmp_path):
    m, o, s = make_model_opt()
    save_checkpoint(tmp_path / "c.pt", m, o, s, epoch=0, global_step=0,
                    best_eval=None, bad_evals=0)
    expected = (random.random(), float(np.random.rand()), torch.rand(1).item())
    load_checkpoint(tmp_path / "c.pt", m, o, s)   # restores RNG to save-time state
    got = (random.random(), float(np.random.rand()), torch.rand(1).item())
    assert expected == got


def test_rotation_keeps_newest(tmp_path):
    m, o, s = make_model_opt()
    for step in (10, 20, 30, 40):
        save_checkpoint(tmp_path / f"ckpt_step_{step}.pt", m, o, s,
                        epoch=0, global_step=step, best_eval=None, bad_evals=0)
    rotate_checkpoints(tmp_path, limit=2)
    kept = sorted(p.name for p in tmp_path.glob("ckpt_step_*.pt"))
    assert kept == ["ckpt_step_30.pt", "ckpt_step_40.pt"]
