import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logbert.config import ModelConfig
from logbert.model import LogBertClassifier

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualize_weights")


def analyze_linear1_weights(model, logger=None, save_path=None, num_bins=50):
    w = model.classifier.linear1.weight.detach().float().view(-1).cpu()

    w_sum = torch.norm(w, 1)
    if logger:
        logger.info(f"[linear1] sum={w_sum:.6f}")

    w_min = w.min().item()
    w_max = w.max().item()
    w_mean = w.mean().item()
    w_std = w.std(unbiased=False).item()
    q = torch.quantile(w, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]))

    if logger:
        logger.info(f"[linear1] shape={tuple(model.classifier.linear1.weight.shape)}")
        logger.info(f"[linear1] min={w_min:.6f}, max={w_max:.6f}, mean={w_mean:.6f}, std={w_std:.6f}")
        logger.info(
            f"[linear1] q0={q[0].item():.6f}, q25={q[1].item():.6f}, "
            f"q50={q[2].item():.6f}, q75={q[3].item():.6f}, q100={q[4].item():.6f}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    w_np = w.numpy()

    n, bins, patches = ax.hist(w_np, bins=num_bins, density=True, alpha=0.75, color="#4C72B0", edgecolor="none")

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(w_np, bw_method="scott")
    xs = np.linspace(w_min, w_max, 300)
    ax.plot(xs, kde(xs), color="#C44E52", linewidth=1.6, label="KDE")

    ax.axvline(w_mean,   color="#DD8452", linewidth=1.2, linestyle="--", label=f"Mean   {w_mean:.4f}")
    ax.axvline(q[2].item(), color="#55A868", linewidth=1.2, linestyle=":",  label=f"Median {q[2].item():.4f}")

    ax.set_title("Weight Distribution — linear1", fontsize=13, pad=10)
    ax.set_xlabel("Weight value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(framealpha=0.5, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        if logger:
            logger.info(f"Saved histogram to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main(model_path: str, output_dir: str):
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model = LogBertClassifier(ModelConfig(**state["config"]))
    model.load_state_dict(state["model"])
    model.eval()
    analyze_linear1_weights(model, logger=logger,
                            save_path=os.path.join(output_dir, "linear1_hist.png"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    main(a.model, a.output_dir)
