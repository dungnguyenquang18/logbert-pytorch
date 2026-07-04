import os

from logbert.data.log_sequences import LogSequenceDataset
from logbert.data.sampler import BucketBatchSampler


def _flat(batches):
    return sorted(i for b in batches for i in b)


def test_covers_all_indices_once(word_vocab, synthetic_sequences):
    ds = LogSequenceDataset(synthetic_sequences, vocab=word_vocab)
    s = BucketBatchSampler(ds, batch_size=16, mode="train", seed=1)
    batches = list(iter(s))
    assert _flat(batches) == list(range(len(ds)))
    assert len(batches) == len(s)


def test_buckets_are_length_homogeneous(word_vocab, synthetic_sequences):
    ds = LogSequenceDataset(synthetic_sequences, vocab=word_vocab)
    s = BucketBatchSampler(ds, batch_size=16, mode="train", seed=1)
    for b in iter(s):
        lens = [ds.lengths[i] for i in b]
        assert max(lens) - min(lens) <= 16  # sorted buckets → tight ranges


def test_train_epochs_shuffle_differently_eval_stable(word_vocab, synthetic_sequences):
    ds = LogSequenceDataset(synthetic_sequences, vocab=word_vocab)
    tr = BucketBatchSampler(ds, batch_size=16, mode="train", seed=1)
    a, b = [x[:] for x in iter(tr)], [x[:] for x in iter(tr)]
    assert a != b  # epoch bump changes order
    ev = BucketBatchSampler(ds, batch_size=16, mode="eval", seed=1)
    c, d = [x[:] for x in iter(ev)], [x[:] for x in iter(ev)]
    assert c == d  # eval deterministic


def test_resume_epoch_reproduces_order(word_vocab, synthetic_sequences):
    ds = LogSequenceDataset(synthetic_sequences, vocab=word_vocab)
    s1 = BucketBatchSampler(ds, batch_size=16, mode="train", seed=1)
    _ = list(iter(s1)); _ = list(iter(s1))          # epochs 1, 2
    third_fresh = [x[:] for x in iter(s1)]          # epoch 3
    s2 = BucketBatchSampler(ds, batch_size=16, mode="train", seed=1, resume_from_epoch=2)
    third_resumed = [x[:] for x in iter(s2)]        # also epoch 3
    assert third_fresh == third_resumed


def test_ddp_slicing_pads_evenly(word_vocab, synthetic_sequences, monkeypatch):
    ds = LogSequenceDataset(synthetic_sequences, vocab=word_vocab)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    r0 = list(iter(BucketBatchSampler(ds, batch_size=16, mode="train", seed=1)))
    monkeypatch.setenv("RANK", "1")
    r1 = list(iter(BucketBatchSampler(ds, batch_size=16, mode="train", seed=1)))
    assert len(r0) == len(r1)  # equal batch counts — the DDP-crash guarantee
