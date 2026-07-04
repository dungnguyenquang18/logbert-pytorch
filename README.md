# logbert-pytorch

Pure-PyTorch reimplementation of the DeviceIncidents LogBERT pipeline (no HuggingFace).

## Setup

Shared venv (Python 3.12, already has torch/pytest): `/home/dungnq/study/5Y1S/meobtnx/.venv`

```bash
source /home/dungnq/study/5Y1S/meobtnx/.venv/bin/activate
```

`activate` only applies to the shell that ran it — if you're scripting single
commands, either activate first in the same line or call the interpreter
directly: `/home/dungnq/study/5Y1S/meobtnx/.venv/bin/python`.

To (re)install dependencies into that venv:
```bash
pip install -r requirements.txt
```

Requirements: Python 3.10+, PyTorch, tensorboard, numpy, pandas, pyarrow,
scikit-learn, matplotlib, scipy, tqdm, pytest.

## Quick smoke test (no real data needed)

Generates small synthetic data in the same PKL/vocab format as the real
pipeline, then runs the whole flow end-to-end in seconds — useful to verify
the code works before pointing it at real data.

```bash
python scripts/make_mock_data.py --out mock_data/ --seed 0
python scripts/train.py --config configs/smoke.py
python scripts/predict.py --config configs/smoke.py --model outputs/smoke/model_final.pt
python scripts/plot_loss.py --history outputs/smoke/history.json --output-dir outputs/smoke/plots/
python scripts/visualize_weights.py --model outputs/smoke/model_final.pt --output-dir outputs/smoke/plots/
```

`mock_data/` and `outputs/` are gitignored — safe to delete and regenerate anytime.

## Running on real data

Copy `configs/default.py` to a new file per experiment and edit `model` /
`data` / `train` in place — this is the whole configuration surface, there
are no CLI flags for hyperparameters.

### 1. Training
Single GPU/CPU:
```bash
python scripts/train.py --config configs/<your_config>.py
```

Multi-GPU DDP via `torchrun`:
```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/<your_config>.py
```

To resume, set `resume_from` in the config's `TrainConfig` to a
`checkpoints/ckpt_step_N.pt` path and rerun.

### 2. Prediction & Evaluation
Runs inference on `data.test_dir` and writes `predictions.parquet` plus
ROC AUC / gain chart / confusion matrix plots next to the model:
```bash
python scripts/predict.py --config configs/<your_config>.py --model outputs/<exp>/model_final.pt
```

### 3. Plot Loss Curves
Plots train/valid loss (total, cls, l1, causal) from `history.json`:
```bash
python scripts/plot_loss.py --history outputs/<exp>/history.json --output-dir outputs/<exp>/plots/
```

### 4. Visualize Weights
Plots the learned weight distribution of `linear1` inside `ClassifierHead`
(the per-position token-influence scores used for root-cause analysis):
```bash
python scripts/visualize_weights.py --model outputs/<exp>/model_final.pt --output-dir outputs/<exp>/plots/
```

## Running Tests
```bash
pytest -v
```
