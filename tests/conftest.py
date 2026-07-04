import random
from collections import Counter

import pytest
from logbert.vocab import WordVocab, DeviceVocab


@pytest.fixture(scope="session")
def word_vocab():
    # 20 fake log templates t0..t19
    return WordVocab(Counter({f"t{i}": 20 - i for i in range(20)}))


@pytest.fixture(scope="session")
def device_vocab():
    return DeviceVocab([f"10.0.0.{i}" for i in range(5)])


@pytest.fixture(scope="session")
def synthetic_sequences(word_vocab, device_vocab):
    """200 tuples in the old PKL format:
    (tokens, [window_end], device_ids, label).
    Label correlates with token distribution so a model CAN learn it:
    label 1 sequences are dominated by high-id tokens."""
    rng = random.Random(0)
    seqs = []
    for i in range(200):
        n = rng.randint(5, 40)
        label = 1 if rng.random() < 0.3 else 0
        lo, hi = (12, 19) if label else (0, 11)
        tokens = [word_vocab.stoi[f"t{rng.randint(lo, hi)}"] for _ in range(n)]
        dev = [device_vocab.to_id(f"10.0.0.{i % 5}")] * n
        seqs.append((tokens, [1700000000 + i * 300], dev, label))
    return seqs
