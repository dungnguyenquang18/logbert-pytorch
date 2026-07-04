# Mock Data + Smoke Train/Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate small mock data in the exact format of the real PKL/vocab artifacts, then run the full pipeline end-to-end (train → predict → plot_loss → visualize_weights) to prove the flow works — no real data needed.

**Architecture:** One committed generator script (`scripts/make_mock_data.py`) writes `mock_data/` (gitignored) mirroring the real layout: `vocab.pkl`, `dev_vocab.pkl`, `train/part_*.pkl`, `valid/`, `test_org/`. One committed smoke config (`configs/smoke.py`) points at it with a tiny model. The four existing scripts are then run unchanged against it.

**Tech Stack:** Python 3.12 (venv `/home/dungnq/study/5Y1S/meobtnx/.venv`), PyTorch, pytest. No new dependencies.

## Global Constraints

- Repo: `/home/dungnq/study/5Y1S/meobtnx/logbert-pytorch/` — all paths below relative to it.
- Python interpreter: `/home/dungnq/study/5Y1S/meobtnx/.venv/bin/python` (`python` is NOT on PATH). Every shell block below assumes you first ran:
  ```bash
  export PY=/home/dungnq/study/5Y1S/meobtnx/.venv/bin/python
  ```
- Git identity is already configured repo-locally (`dugnam18@gmail.com` / `dungnq`) — do not change it.
- PKL tuple format (must match `logbert/data/log_sequences.py:_normalize_tuple`): `(tokens: list[int], window: list[int], device_ids: list[int], label: int)`. Tokens are vocab **ids** (ints), device_ids are DeviceVocab ids, window holds one unix-seconds timestamp, label ∈ {0, 1}.
- Mock data and outputs must NOT be committed: `.gitignore` already covers `outputs/`, `*.pkl`, `*.pt`; Task 2 adds `mock_data/` explicitly.
- CPU-only is fine: `Trainer` auto-disables bf16 autocast off-CUDA; everything below runs in seconds on CPU.

---

### Task 1: Mock data generator

**Files:**
- Create: `scripts/make_mock_data.py`
- Test: `tests/test_make_mock_data.py`

**Interfaces:**
- Produces: `generate(out: str, n_train=600, n_valid=200, n_test=200, n_templates=50, n_devices=8, seed=0) -> None` — writes `vocab.pkl`, `dev_vocab.pkl`, `train/part_0.pkl` + `train/part_1.pkl`, `valid/part_0.pkl`, `test_org/part_0.pkl` under `out`. Also runnable as `$PY scripts/make_mock_data.py --out mock_data/ --seed 0`.
- Consumes: `WordVocab`/`DeviceVocab` from `logbert/vocab.py` (their `save_vocab` pickles are loadable by the scripts' `load_vocab`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_make_mock_data.py`:

```python
import torch

from logbert.data.log_sequences import LogCollator, LogSequenceDataset, load_pkl_dir
from logbert.vocab import DeviceVocab, WordVocab
from scripts.make_mock_data import generate


def test_generate_creates_loadable_artifacts(tmp_path):
    generate(str(tmp_path), n_train=40, n_valid=10, n_test=10,
             n_templates=20, n_devices=4, seed=1)

    vocab = WordVocab.load_vocab(str(tmp_path / "vocab.pkl"))
    dev_vocab = DeviceVocab.load_vocab(str(tmp_path / "dev_vocab.pkl"))
    assert len(vocab) == 25          # 20 templates + 5 specials
    assert len(dev_vocab) == 6       # 4 devices + 2 specials

    train = load_pkl_dir(str(tmp_path / "train"))
    assert len(train) == 40
    tokens, window, dev, label = train[0]
    assert all(isinstance(t, int) for t in tokens)
    assert all(5 <= t < len(vocab) for t in tokens)      # ids past the specials
    assert len(dev) == len(tokens)
    assert len(window) == 1 and window[0] > 1_000_000_000
    assert label in (0, 1)
    assert len(load_pkl_dir(str(tmp_path / "valid"))) == 10
    assert len(load_pkl_dir(str(tmp_path / "test_org"))) == 10

    # the whole thing must flow through Dataset + Collator
    ds = LogSequenceDataset(train, vocab=vocab)
    batch = LogCollator(vocab=vocab, seq_len=64)([ds[i] for i in range(8)])
    assert batch["input_ids"].shape == batch["mlm_labels"].shape
    assert batch["labels"].dtype == torch.long and batch["labels"].shape == (8,)


def test_generate_writes_two_train_parts_and_both_labels(tmp_path):
    generate(str(tmp_path), n_train=40, n_valid=10, n_test=10,
             n_templates=20, n_devices=4, seed=1)
    assert (tmp_path / "train" / "part_0.pkl").exists()
    assert (tmp_path / "train" / "part_1.pkl").exists()
    labels = [t[3] for t in load_pkl_dir(str(tmp_path / "train"))]
    assert 0 in labels and 1 in labels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/dungnq/study/5Y1S/meobtnx/logbert-pytorch && $PY -m pytest tests/test_make_mock_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.make_mock_data'`

- [ ] **Step 3: Write the generator**

Create `scripts/make_mock_data.py`:

```python
"""Generate small mock artifacts mirroring the real pipeline inputs.

Writes under --out (default mock_data/):
    vocab.pkl        WordVocab over fake templates t0..t{n_templates-1}
    dev_vocab.pkl    DeviceVocab over fake device IPs 10.0.0.*
    train/part_0.pkl, train/part_1.pkl
    valid/part_0.pkl
    test_org/part_0.pkl

Each PKL part is a list of tuples in the old-repo format:
    (tokens: list[int], [window_end: int], device_ids: list[int], label: int)

The label is learnable: label-1 sequences draw from the high-id template
range, label-0 from the low-id range — same trick as tests/conftest.py.
"""
import argparse
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logbert.vocab import DeviceVocab, WordVocab


def make_vocabs(n_templates: int, n_devices: int):
    word_vocab = WordVocab(Counter({f"t{i}": n_templates - i for i in range(n_templates)}))
    device_vocab = DeviceVocab([f"10.0.0.{i}" for i in range(n_devices)])
    return word_vocab, device_vocab


def make_sequences(n, word_vocab, device_vocab, rng, pos_ratio=0.3,
                   min_len=5, max_len=60, t0=1_700_000_000):
    n_templates = len(word_vocab) - 5                 # minus the 5 specials
    n_devices = len(device_vocab) - 2                 # minus the 2 specials
    split = n_templates * 6 // 10
    seqs = []
    for i in range(n):
        length = rng.randint(min_len, max_len)
        label = 1 if rng.random() < pos_ratio else 0
        lo, hi = (split, n_templates - 1) if label else (0, split - 1)
        tokens = [word_vocab.stoi[f"t{rng.randint(lo, hi)}"] for _ in range(length)]
        dev = [device_vocab.to_id(f"10.0.0.{i % n_devices}")] * length
        seqs.append((tokens, [t0 + i * 300], dev, label))
    return seqs


def write_parts(seqs, out_dir: Path, n_parts: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    per = (len(seqs) + n_parts - 1) // n_parts
    for p in range(n_parts):
        with open(out_dir / f"part_{p}.pkl", "wb") as f:
            pickle.dump(seqs[p * per:(p + 1) * per], f)


def generate(out: str, n_train=600, n_valid=200, n_test=200,
             n_templates=50, n_devices=8, seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    word_vocab, device_vocab = make_vocabs(n_templates, n_devices)
    word_vocab.save_vocab(str(out / "vocab.pkl"))
    device_vocab.save_vocab(str(out / "dev_vocab.pkl"))
    write_parts(make_sequences(n_train, word_vocab, device_vocab, rng), out / "train", 2)
    write_parts(make_sequences(n_valid, word_vocab, device_vocab, rng), out / "valid", 1)
    write_parts(make_sequences(n_test, word_vocab, device_vocab, rng), out / "test_org", 1)
    print(f"mock data written to {out.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="mock_data/")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    generate(a.out, seed=a.seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/dungnq/study/5Y1S/meobtnx/logbert-pytorch && $PY -m pytest tests/test_make_mock_data.py -v`
Expected: 2 passed

Also run the full suite to check nothing broke: `$PY -m pytest tests/ -q`
Expected: 40 passed

- [ ] **Step 5: Commit**

```bash
cd /home/dungnq/study/5Y1S/meobtnx/logbert-pytorch
git add scripts/make_mock_data.py tests/test_make_mock_data.py
git commit -m "feat: mock data generator mirroring real PKL/vocab artifacts"
```

---

### Task 2: Smoke config + end-to-end pipeline run

**Files:**
- Create: `configs/smoke.py`
- Modify: `.gitignore` (add `mock_data/`)

**Interfaces:**
- Consumes: `generate()` from Task 1 (via CLI); `scripts/train.py`, `scripts/predict.py`, `scripts/plot_loss.py`, `scripts/visualize_weights.py` — all unchanged.
- Produces: `configs/smoke.py` defining module-level `model`, `data`, `train` (the contract `load_config_module` checks).

- [ ] **Step 1: Write the smoke config**

Create `configs/smoke.py`:

```python
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
    max_seq_len=64, causal=True, use_mlm=True, use_l1=True,
    alpha_mlm=0.1, lambda_l1=1e-4, is_device=False,
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
    epochs=3, lr=1e-3, grad_accum_steps=2, bf16=True,
    log_every=5, eval_every=0, save_every=0, save_total_limit=2,
    early_stop_patience=10, min_delta=0.0,
)
```

Note: `max_seq_len=64` must stay ≥ generator `max_len` (60) + 1 SOS, and `data.seq_len` caps the collator at the same 64.

- [ ] **Step 2: Add `mock_data/` to .gitignore**

Append one line to `.gitignore`:

```
mock_data/
```

- [ ] **Step 3: Generate the mock data**

Run:
```bash
cd /home/dungnq/study/5Y1S/meobtnx/logbert-pytorch
$PY scripts/make_mock_data.py --out mock_data/ --seed 0
```
Expected: `mock data written to /home/dungnq/study/5Y1S/meobtnx/logbert-pytorch/mock_data` and the tree `mock_data/{vocab.pkl,dev_vocab.pkl,train/part_0.pkl,train/part_1.pkl,valid/part_0.pkl,test_org/part_0.pkl}` exists. `git status --short` shows only `configs/smoke.py`, `.gitignore` — no mock_data files.

- [ ] **Step 4: Run training**

Run:
```bash
cd /home/dungnq/study/5Y1S/meobtnx/logbert-pytorch
$PY scripts/train.py --config configs/smoke.py
```
Expected (finishes in well under a minute on CPU):
- log lines `vocab=55 device_vocab=10`, `train=600 eval=200`, `class weights: [...]`, `parameters: ...`
- `outputs/smoke/history.json` exists; contains exactly one entry with `initial_train_loss`/`initial_eval_loss` at step 0, three `eval_loss` entries (one per epoch, since `eval_every=0`), train entries carrying `loss`, `learning_rate`, `train_loss_cls`, `train_loss_mlm`, `train_loss_l1`
- `outputs/smoke/model_final.pt` exists; `outputs/smoke/checkpoints/` has ≤ 2 `ckpt_step_*.pt` (rotation)
- `outputs/smoke/tb_logs/` has a TensorBoard event file

Verify learning happened:
```bash
$PY - <<'EOF'
import json
h = json.load(open("outputs/smoke/history.json"))
init = next(e for e in h if "initial_eval_loss" in e)["initial_eval_loss"]
finals = [e["eval_loss"] for e in h if "eval_loss" in e]
print("initial:", init, "evals:", finals)
assert finals[-1] < init, "eval loss did not improve"
EOF
```
Expected: prints losses; last `eval_loss` < initial. (Mock labels are strongly separable — it should drop clearly.)

- [ ] **Step 5: Run predict**

Run:
```bash
cd /home/dungnq/study/5Y1S/meobtnx/logbert-pytorch
$PY scripts/predict.py --config configs/smoke.py --model outputs/smoke/model_final.pt
```
Expected:
- log line `TP=... FP=... TN=... FN=... | P=... R=... F1=...` with F1 clearly above 0.8 (separable mock data)
- `outputs/smoke/predictions.parquet` exists with columns `ip`, `timestamp`, `pred_proba`, `labels`
- `outputs/smoke/roc_auc/`, `outputs/smoke/gain_chart/`, `outputs/smoke/conf_matrix/` each contain a PNG

- [ ] **Step 6: Run the two plot scripts**

Run:
```bash
cd /home/dungnq/study/5Y1S/meobtnx/logbert-pytorch
$PY scripts/plot_loss.py --history outputs/smoke/history.json --output-dir outputs/smoke/plots/
$PY scripts/visualize_weights.py --model outputs/smoke/model_final.pt --output-dir outputs/smoke/plots/
```
Expected: both exit 0; `outputs/smoke/plots/` contains the loss-curve PNGs (total/cls/l1/mlm) and the linear1 weight plots.

- [ ] **Step 7: Full test suite still green, then commit**

Run: `$PY -m pytest tests/ -q`
Expected: 40 passed

```bash
cd /home/dungnq/study/5Y1S/meobtnx/logbert-pytorch
git add configs/smoke.py .gitignore
git commit -m "feat: smoke config for end-to-end mock-data run"
```

---

## Verification (whole plan)

1. `tests/test_make_mock_data.py` green — mock artifacts load through the same code paths as real data (`load_vocab`, `load_pkl_dir`, Dataset, Collator).
2. Train run on mock data: `eval_loss` at epoch 3 < initial eval loss; `history.json` schema carries all loss components; checkpoint rotation capped at 2.
3. Predict run: parquet + ROC/gain/CM artifacts produced; F1 high on separable data.
4. Both plot scripts consume the run's outputs without error.
5. `git status` clean of any `mock_data/` or `outputs/` files after commits.
