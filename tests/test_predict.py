import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_predict_end_to_end(tmp_path, word_vocab, device_vocab, synthetic_sequences):
    from logbert.config import ModelConfig
    from logbert.model import LogBertClassifier

    # save a tiny trained-ish model
    mcfg = ModelConfig(vocab_size=len(word_vocab), hidden=32, layers=1, attn_heads=2,
                       max_seq_len=64, num_devices=len(device_vocab))
    torch.manual_seed(0)
    m = LogBertClassifier(mcfg)
    model_path = tmp_path / "model_final.pt"
    torch.save({"model": m.state_dict(), "config": vars(mcfg)}, model_path)

    data_dir = tmp_path / "data"; (data_dir / "test").mkdir(parents=True)
    with open(data_dir / "test" / "p.pkl", "wb") as f:
        pickle.dump(synthetic_sequences[:64], f)
    word_vocab.save_vocab(str(data_dir / "vocab.pkl"))
    device_vocab.save_vocab(str(data_dir / "dev_vocab.pkl"))

    cfg_file = tmp_path / "exp.py"
    cfg_file.write_text(f"""
from logbert.config import ModelConfig, DataConfig, TrainConfig
model = ModelConfig(vocab_size=0, hidden=32, layers=1, attn_heads=2, max_seq_len=64)
data = DataConfig(train_dir="unused", test_dir=r"{data_dir / 'test'}",
                  vocab_path=r"{data_dir / 'vocab.pkl'}",
                  device_vocab_path=r"{data_dir / 'dev_vocab.pkl'}",
                  seq_len=64, batch_size=16, num_workers=0)
train = TrainConfig(output_dir=r"{tmp_path}")
""")
    import predict as predict_script
    df = predict_script.run_predict(str(cfg_file), str(model_path))

    assert set(df.columns) == {"ip", "timestamp", "pred_proba", "labels"}
    assert len(df) == 64
    out_dir = model_path.parent
    assert (out_dir / "predictions.parquet").exists()
    assert list((out_dir / "roc_auc").glob("*.png"))
    assert list((out_dir / "conf_matrix").glob("*.png"))
