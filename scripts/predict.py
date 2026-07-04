"""Evaluate a trained model on data.test_dir and emit predictions + charts.
Usage: python scripts/predict.py --config configs/default.py --model outputs/default/model_final.pt
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logbert.config import ModelConfig
from logbert.data.log_sequences import LogCollator, LogSequenceDataset, load_pkl_dir
from logbert.data.sampler import BucketBatchSampler
from logbert.metrics import plot_cm, return_percentile_gain_chart, visualize_roc_auc
from logbert.model import LogBertClassifier
from logbert.vocab import DeviceVocab, WordVocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("predict")
VNTZ = pytz.timezone("Asia/Ho_Chi_Minh")


def load_model(model_path: str, cfg=None) -> LogBertClassifier:
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if "optimizer" in state:
        if cfg is not None:
            model_cfg = cfg.model
        else:
            model_cfg = ModelConfig(**state["config"])
    else:
        model_cfg = ModelConfig(**state["config"])
    
    model = LogBertClassifier(model_cfg)
    model.load_state_dict(state["model"])
    return model


def run_predict(config_path: str, model_path: str) -> pd.DataFrame:
    sys.path.insert(0, str(Path(config_path).resolve().parent))
    # Import train script helper
    from train import load_config_module
    cfg = load_config_module(config_path)
    data_cfg = cfg.data

    vocab = WordVocab.load_vocab(data_cfg.vocab_path)
    device_vocab = DeviceVocab.load_vocab(data_cfg.device_vocab_path)

    # We fill vocab_size dynamically before loading the model config
    cfg.model.vocab_size = len(vocab)
    cfg.model.num_devices = len(device_vocab)

    model = load_model(model_path, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    seqs = load_pkl_dir(data_cfg.test_dir)
    ds = LogSequenceDataset(seqs, vocab=vocab, predict_mode=True)
    loader = DataLoader(ds,
                        batch_sampler=BucketBatchSampler(ds, batch_size=data_cfg.batch_size,
                                                         seed=42, mode="eval"),
                        collate_fn=LogCollator(vocab=vocab, seq_len=data_cfg.seq_len),
                        num_workers=data_cfg.num_workers,
                        pin_memory=torch.cuda.is_available())

    tp = fp = tn = fn = 0
    preds_all, gts_all, probs_all, records = [], [], [], []
    total_cls, num_batch = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            input_ids = batch["input_ids"].to(device)
            device_ids = batch["device_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, device_ids=device_ids, labels=labels)
            probs = torch.softmax(out["logits"], dim=-1)[:, 1]
            pred = out["logits"].argmax(dim=-1)

            valid = labels >= 0
            pv, gv = pred[valid], labels[valid]
            tp += int(((pv == 1) & (gv == 1)).sum())
            fp += int(((pv == 1) & (gv == 0)).sum())
            tn += int(((pv == 0) & (gv == 0)).sum())
            fn += int(((pv == 0) & (gv == 1)).sum())
            preds_all += pv.int().cpu().tolist()
            gts_all += gv.int().cpu().tolist()
            probs_all += probs[valid].cpu().tolist()
            if out["loss_cls"] is not None:
                total_cls += out["loss_cls"].item(); num_batch += 1

            for dev_row, ts, prob, lab in zip(batch["device_ids"].tolist(),
                                              batch["window_end"].tolist(),
                                              probs.cpu().tolist(),
                                              labels.cpu().tolist()):
                ip = device_vocab.itos[dev_row[0]] if dev_row else "unknown"
                records.append({
                    "ip": ip,
                    "timestamp": pd.to_datetime(ts, unit="s", utc=True)
                                   .tz_convert("Asia/Ho_Chi_Minh").strftime("%Y-%m-%d %H:%M:%S"),
                    "pred_proba": float(prob),
                    "labels": int(lab),
                })

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    logger.info(f"TP={tp} FP={fp} TN={tn} FN={fn} | P={prec:.4f} R={rec:.4f} F1={f1:.4f}")
    if num_batch:
        logger.info(f"cls loss: {total_cls / num_batch:.6f}")

    out_dir = Path(model_path).parent
    df = pd.DataFrame(records, columns=["ip", "timestamp", "pred_proba", "labels"])
    df.to_parquet(out_dir / "predictions.parquet", index=False)

    df_test = pd.DataFrame({"pred": preds_all, "gt": gts_all, "prob": probs_all})
    visualize_roc_auc(np.array(gts_all), np.array(probs_all), save_fig=True,
                      output_dir=str(out_dir / "roc_auc"))
    return_percentile_gain_chart(df_test, true_col="gt", y_pred="pred", y_proba="prob",
                                 number_of_thresholds=10, save_fig=True,
                                 output_dir=str(out_dir / "gain_chart"),
                                 plot_name="(LogBERT) Precision-Coverage by Decile")
    plot_cm(df_test["gt"], df_test["pred"],
            out=str(out_dir / "conf_matrix" /
                    f"pred_cm_{datetime.now(VNTZ).strftime('%d-%m-%y_%H-%M')}.png"))
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.py")
    p.add_argument("--model", required=True)
    run_predict(p.parse_args().config, p.parse_args().model)
