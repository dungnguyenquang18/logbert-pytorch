import torch

from logbert.data.log_sequences import LogCollator, LogSequenceDataset, load_pkl_dir
from logbert.vocab import DeviceVocab, WordVocab
from scripts.make_mock_data import generate


def test_generate_creates_loadable_artifacts(tmp_path):
    generate(str(tmp_path), n_train=40, n_valid=10, n_test=10,
             n_templates=20, n_devices=4, seed=1)

    vocab = WordVocab.load_vocab(str(tmp_path / "vocab.pkl"))
    dev_vocab = DeviceVocab.load_vocab(str(tmp_path / "dev_vocab.pkl"))
    assert len(vocab) == 25          # 20 templates + 5 specials
    assert len(dev_vocab) == 6       # 4 devices + 2 specials

    train = load_pkl_dir(str(tmp_path / "train"))
    assert len(train) == 40
    tokens, window, dev, label = train[0]
    assert all(isinstance(t, int) for t in tokens)
    assert all(5 <= t < len(vocab) for t in tokens)      # ids past the specials
    assert len(dev) == len(tokens)
    assert len(window) == 1 and window[0] > 1_000_000_000
    assert label in (0, 1)
    assert len(load_pkl_dir(str(tmp_path / "valid"))) == 10
    assert len(load_pkl_dir(str(tmp_path / "test_org"))) == 10

    # the whole thing must flow through Dataset + Collator
    ds = LogSequenceDataset(train, vocab=vocab)
    batch = LogCollator(vocab=vocab, seq_len=64)([ds[i] for i in range(8)])
    assert batch["input_ids"].shape == batch["mlm_labels"].shape
    assert batch["labels"].dtype == torch.long and batch["labels"].shape == (8,)


def test_generate_writes_two_train_parts_and_both_labels(tmp_path):
    generate(str(tmp_path), n_train=40, n_valid=10, n_test=10,
             n_templates=20, n_devices=4, seed=1)
    assert (tmp_path / "train" / "part_0.pkl").exists()
    assert (tmp_path / "train" / "part_1.pkl").exists()
    labels = [t[3] for t in load_pkl_dir(str(tmp_path / "train"))]
    assert 0 in labels and 1 in labels
