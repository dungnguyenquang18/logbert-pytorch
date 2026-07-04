import torch

from logbert.data.log_sequences import LogSequenceDataset, LogCollator


def test_getitem_prepends_sos_and_aligns_device(word_vocab, device_vocab, synthetic_sequences):
    ds = LogSequenceDataset(synthetic_sequences, vocab=word_vocab)
    raw_tokens, window, raw_dev, label = synthetic_sequences[0]
    s = ds[0]
    assert s["tokens"][0] == word_vocab.sos_index
    assert len(s["tokens"]) == len(raw_tokens) + 1
    assert s["tokens"][1:] == raw_tokens          # ints pass through unmasked
    assert s["device_ids"][0] == raw_dev[0]       # dev_cls prepended
    assert len(s["device_ids"]) == len(s["tokens"])
    assert s["label"] == label and s["window_end"] == window[0]
    assert ds.lengths[0] == len(raw_tokens) + 1


def test_short_tuple_variants(word_vocab):
    ds = LogSequenceDataset([([5, 6], [7])], vocab=word_vocab)  # 2-tuple: (tokens, dev)
    s = ds[0]
    assert s["label"] == -1 and s["window_end"] == 0


def test_collator_causal_labels_and_padding(word_vocab, synthetic_sequences):
    ds = LogSequenceDataset(synthetic_sequences, vocab=word_vocab)
    coll = LogCollator(vocab=word_vocab, seq_len=1000)
    samples = [ds[i] for i in range(4)]
    batch = coll(samples)
    L = max(len(s["tokens"]) for s in samples)
    assert batch["input_ids"].shape == (4, L)
    for key in ("input_ids", "causal_labels", "device_ids"):
        assert batch[key].shape == (4, L) and batch[key].dtype == torch.long
    assert batch["labels"].shape == (4,) and batch["window_end"].shape == (4,)
    # causal shift: label[i] == input[i+1] on valid positions
    n0 = len(samples[0]["tokens"])
    assert batch["causal_labels"][0, : n0 - 1].tolist() == batch["input_ids"][0, 1:n0].tolist()
    assert batch["causal_labels"][0, n0 - 1].item() == word_vocab.eos_index
    if n0 < L:  # padding is 0
        assert batch["causal_labels"][0, n0:].sum().item() == 0
        assert batch["input_ids"][0, n0:].sum().item() == 0


def test_collator_caps_at_seq_len(word_vocab):
    long_seq = (list(range(5, 105)), [0], [1] * 100, 0)   # 100 tokens
    ds = LogSequenceDataset([long_seq], vocab=word_vocab)
    coll = LogCollator(vocab=word_vocab, seq_len=50)
    batch = coll([ds[0]])
    assert batch["input_ids"].shape == (1, 50)


def test_load_pkl_dir(tmp_path, synthetic_sequences):
    import pickle
    (tmp_path / "sub").mkdir()
    with open(tmp_path / "a.pkl", "wb") as f:
        pickle.dump(synthetic_sequences[:100], f)
    with open(tmp_path / "sub" / "b.pkl", "wb") as f:
        pickle.dump(synthetic_sequences[100:], f)
    from logbert.data.log_sequences import load_pkl_dir
    assert len(load_pkl_dir(str(tmp_path))) == 200
