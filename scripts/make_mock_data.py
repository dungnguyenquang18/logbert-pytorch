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
