#!/usr/bin/env bash
# Smoke test: mock data -> train -> predict -> plot loss -> visualize pooling weights.
# Run from repo root: bash run_smoke.sh
set -euo pipefail

cd /home/dungnq/study/5Y1S/meobtnx/logbert-pytorch
source /home/dungnq/study/5Y1S/meobtnx/.venv/bin/activate

# python scripts/make_mock_data.py --out mock_data/

python scripts/train.py --config configs/smoke.py

python scripts/predict.py --config configs/smoke.py --model outputs/smoke/model_final.pt

python scripts/plot_loss.py --history outputs/smoke/history.json --output-dir outputs/smoke/plots/

# python scripts/visualize_weights.py --model outputs/smoke/model_final.pt --output-dir outputs/smoke/plots/

echo "Done. Artifacts under outputs/smoke/"
