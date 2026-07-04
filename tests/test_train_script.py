import json
import pickle
import sys
from pathlib import Path

import torch


def test_train_script_end_to_end(tmp_path, word_vocab, device_vocab, synthetic_sequences):
    """Full wiring test: write pkl+vocab to disk, generate a config file,
    run scripts/train.py main(), check artifacts."""
    data_dir = tmp_path / "data"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "valid").mkdir()
    with open(data_dir / "train" / "part0.pkl", "wb") as f:
        pickle.dump(synthetic_sequences[:160], f)
    with open(data_dir / "valid" / "part0.pkl", "wb") as f:
        pickle.dump(synthetic_sequences[160:], f)
    word_vocab.save_vocab(str(data_dir / "vocab.pkl"))
    device_vocab.save_vocab(str(data_dir / "dev_vocab.pkl"))

    out_dir = tmp_path / "out"
    cfg_file = tmp_path / "exp.py"
    cfg_file.write_text(f"""
from logbert.config import ModelConfig, DataConfig, TrainConfig
model = ModelConfig(vocab_size=0, hidden=32, layers=1, attn_heads=2, max_seq_len=64)
data = DataConfig(train_dir=r"{data_dir / 'train'}", valid_dir=r"{data_dir / 'valid'}",
                  vocab_path=r"{data_dir / 'vocab.pkl'}",
                  device_vocab_path=r"{data_dir / 'dev_vocab.pkl'}",
                  seq_len=64, batch_size=16, num_workers=0)
train = TrainConfig(output_dir=r"{out_dir}", epochs=1, lr=1e-3, grad_accum_steps=2,
                    bf16=False, log_every=2, log_initial_loss=False,
                    cuda_clear_each_epoch=False)
""")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train as train_script
    train_script.main(str(cfg_file))

    hist = json.loads((out_dir / "history.json").read_text())
    assert any("eval_loss" in h for h in hist)
    final = torch.load(out_dir / "model_final.pt", map_location="cpu", weights_only=False)
    assert final["config"]["vocab_size"] == 25   # 20 tokens + 5 specials (filled at runtime)


def test_calc_class_weight():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train as train_script
    w = train_script.calc_class_weight([0, 0, 0, 1])
    assert w.shape == (2,) and w[1] > w[0]
    assert torch.equal(train_script.calc_class_weight([]), torch.ones(2))
