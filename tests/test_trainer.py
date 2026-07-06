import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from logbert.config import ModelConfig, TrainConfig
from logbert.data.log_sequences import LogCollator, LogSequenceDataset
from logbert.data.sampler import BucketBatchSampler
from logbert.model import LogBertClassifier
from logbert.training.trainer import Trainer


def make_loaders(word_vocab, synthetic_sequences, batch_size=16):
    ds = LogSequenceDataset(synthetic_sequences, vocab=word_vocab)
    coll = LogCollator(vocab=word_vocab, seq_len=64)
    def loader(mode, resume_epoch=0):
        return DataLoader(ds, batch_sampler=BucketBatchSampler(
            ds, batch_size=batch_size, mode=mode, seed=42, resume_from_epoch=resume_epoch),
            collate_fn=coll, num_workers=0)
    return loader("train"), loader("eval")


def make_model(word_vocab):
    torch.manual_seed(42)
    return LogBertClassifier(ModelConfig(vocab_size=len(word_vocab), hidden=32, layers=1,
                                         attn_heads=2, max_seq_len=64, causal=True,
                                         use_causal_lm=True, is_device=False,
                                         num_devices=6))


def base_cfg(tmp_path, **over):
    d = dict(output_dir=str(tmp_path / "out"), epochs=2, lr=1e-3, grad_accum_steps=2,
             bf16=False, seed=42, log_every=2, eval_every=0, save_every=0,
             save_total_limit=3, early_stop_patience=10, min_delta=0.0,
             warmup_ratio=0.1, log_initial_loss=True, cuda_clear_each_epoch=False,
             scale_lr_with_world_size=False)
    d.update(over)
    return TrainConfig(**d)


def test_fit_decreases_loss_and_writes_artifacts(tmp_path, word_vocab, synthetic_sequences):
    train_dl, eval_dl = make_loaders(word_vocab, synthetic_sequences)
    t = Trainer(make_model(word_vocab), train_dl, eval_dl, base_cfg(tmp_path))
    t.fit()
    out = Path(t.cfg.output_dir)
    hist = json.loads((out / "history.json").read_text())
    initial = [h for h in hist if "initial_train_loss" in h]
    evals = [h for h in hist if "eval_loss" in h]
    assert len(initial) == 1 and initial[0]["step"] == 0
    assert len(evals) == 2                          # eval_every=0 → once per epoch
    assert evals[-1]["eval_loss"] < initial[0]["initial_eval_loss"]  # it learned
    assert "eval_loss_cls" in evals[-1]
    assert (out / "model_final.pt").exists()
    assert len(list((out / "checkpoints").glob("ckpt_step_*.pt"))) >= 1


def test_model_final_loads_into_fresh_model_when_class_weight_was_set(tmp_path, word_vocab,
                                                                       synthetic_sequences):
    """class_weight is a training-only buffer (registered as None by default, so it's
    absent from a fresh model's state_dict). If set_class_weight() was called before
    training, model_final.pt must still load cleanly into a model that never had it set —
    this is exactly what scripts/predict.py does."""
    train_dl, eval_dl = make_loaders(word_vocab, synthetic_sequences)
    model = make_model(word_vocab)
    model.set_class_weight(torch.tensor([0.5, 1.5]))
    t = Trainer(model, train_dl, eval_dl, base_cfg(tmp_path))
    t.fit()

    state = torch.load(Path(t.cfg.output_dir) / "model_final.pt",
                       map_location="cpu", weights_only=False)
    fresh = make_model(word_vocab)
    fresh.load_state_dict(state["model"])  # must not raise on "class_weight"


def test_resume_matches_uninterrupted_run(tmp_path, word_vocab, synthetic_sequences):
    # Run A: 4 epochs straight
    train_dl, eval_dl = make_loaders(word_vocab, synthetic_sequences)
    ta = Trainer(make_model(word_vocab), train_dl, eval_dl,
                 base_cfg(tmp_path / "a", epochs=4, log_initial_loss=False,
                          scheduler_type="constant_with_warmup", warmup_ratio=0.0))
    ta.fit()
    # Run B: 2 epochs, then resume for 2 more
    train_dl, eval_dl = make_loaders(word_vocab, synthetic_sequences)
    tb = Trainer(make_model(word_vocab), train_dl, eval_dl,
                 base_cfg(tmp_path / "b", epochs=2, log_initial_loss=False,
                          scheduler_type="constant_with_warmup", warmup_ratio=0.0))
    tb.fit()
    ckpts = sorted((Path(tb.cfg.output_dir) / "checkpoints").glob("ckpt_step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    train_dl, eval_dl = make_loaders(word_vocab, synthetic_sequences)
    tc = Trainer(make_model(word_vocab), train_dl, eval_dl,
                 base_cfg(tmp_path / "c", epochs=4, log_initial_loss=False,
                          resume_from=str(ckpts[-1]),
                          scheduler_type="constant_with_warmup", warmup_ratio=0.0))
    tc.fit()
    ha = json.loads((Path(ta.cfg.output_dir) / "history.json").read_text())
    hc = json.loads((Path(tc.cfg.output_dir) / "history.json").read_text())
    a_final = [h for h in ha if "eval_loss" in h][-1]["eval_loss"]
    c_final = [h for h in hc if "eval_loss" in h][-1]["eval_loss"]
    assert a_final == pytest.approx(c_final, rel=1e-4)



def test_early_stopping_stops(tmp_path, word_vocab, synthetic_sequences):
    train_dl, eval_dl = make_loaders(word_vocab, synthetic_sequences)
    # min_delta huge → nothing ever counts as improvement after the first eval
    t = Trainer(make_model(word_vocab), train_dl, eval_dl,
                 base_cfg(tmp_path, epochs=10, early_stop_patience=2, min_delta=1e9,
                          log_initial_loss=False))
    t.fit()
    hist = json.loads((Path(t.cfg.output_dir) / "history.json").read_text())
    evals = [h for h in hist if "eval_loss" in h]
    assert len(evals) == 3                # first eval sets best, then 2 bad → stop


def test_checkpoint_rotation(tmp_path, word_vocab, synthetic_sequences):
    train_dl, eval_dl = make_loaders(word_vocab, synthetic_sequences)
    t = Trainer(make_model(word_vocab), train_dl, eval_dl,
                base_cfg(tmp_path, epochs=4, save_total_limit=2, log_initial_loss=False))
    t.fit()
    ckpts = list((Path(t.cfg.output_dir) / "checkpoints").glob("ckpt_step_*.pt"))
    assert len(ckpts) <= 2
