import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_plot_loss(tmp_path):
    hist = [
        {"step": 0, "epoch": 0, "initial_train_loss": 2.0, "initial_eval_loss": 2.1},
        {"step": 10, "epoch": 1.0, "loss": 1.5, "learning_rate": 1e-4,
         "train_loss_cls": 1.2, "train_loss_l1": 3.0, "train_loss_causal": 0.3},
        {"step": 10, "epoch": 1.0, "eval_loss": 1.6, "eval_loss_cls": 1.3,
         "eval_loss_l1": 3.0, "eval_loss_causal": 0.3},
    ]
    hp = tmp_path / "history.json"
    hp.write_text(json.dumps(hist))
    import plot_loss
    plot_loss.main(str(hp), str(tmp_path / "plots"))
    for name in ("total_loss.png", "cls_loss.png", "l1_loss.png", "causal_loss.png"):
        assert (tmp_path / "plots" / name).exists()


def test_visualize_weights(tmp_path, word_vocab):
    from logbert.config import ModelConfig
    from logbert.model import LogBertClassifier
    mcfg = ModelConfig(vocab_size=len(word_vocab), hidden=32, layers=1,
                       attn_heads=2, max_seq_len=64)
    m = LogBertClassifier(mcfg)
    mp = tmp_path / "model_final.pt"
    torch.save({"model": m.state_dict(), "config": vars(mcfg)}, mp)
    import visualize_weights
    visualize_weights.main(str(mp), str(tmp_path / "w"))
    assert (tmp_path / "w" / "linear1_hist.png").exists()
