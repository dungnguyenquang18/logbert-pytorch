# logbert-pytorch

Pure-PyTorch reimplementation of the DeviceIncidents LogBERT pipeline (no HuggingFace).

## Requirements
* Python 3.10+
* PyTorch
* tensorboard, numpy, pandas, pyarrow, scikit-learn, matplotlib, scipy, tqdm, pytest

To install dependencies:
```bash
pip install -r requirements.txt
```

## Running Scripts

### 1. Training
To train the model on a single GPU/CPU:
```bash
python scripts/train.py --config configs/default.py
```

To train with multi-GPU DDP using `torchrun`:
```bash
torchrun --nproc_per_node=2 scripts/train.py --config configs/default.py
```

### 2. Prediction & Evaluation
To run inference and compute metrics (ROC AUC, Confusion Matrix, Gain Chart) on the test parquet data:
```bash
python scripts/predict.py --config configs/default.py --model outputs/default/model_final.pt
```

### 3. Plot Loss Curves
To plot the train and validation loss components from `history.json`:
```bash
python scripts/plot_loss.py --history outputs/default/history.json --output-dir outputs/default/plots/
```

### 4. Visualize Weights
To plot the learned weights distribution of `linear1` inside the `ClassifierHead`:
```bash
python scripts/visualize_weights.py --model outputs/default/model_final.pt --output-dir outputs/default/plots/
```

## Running Tests
Run the test suite using `pytest`:
```bash
pytest -v
```
