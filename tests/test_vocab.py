import pickle
import sys
import types
from collections import Counter

from logbert.vocab import WordVocab, DeviceVocab


def _make_word_vocab():
    return WordVocab(Counter({"a": 5, "b": 3, "c": 1}))


class DummyWordVocab(WordVocab):
    pass


def test_word_vocab_specials():
    v = _make_word_vocab()
    assert v.itos[:5] == ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"]
    assert v.pad_index == 0 and v.sos_index == 3
    assert len(v) == 8  # 5 specials + a, b, c


def test_device_vocab():
    dv = DeviceVocab(["10.0.0.1", "10.0.0.2", "10.0.0.1"])
    assert dv.itos[:2] == ["<pad>", "<unk>"]
    assert dv.to_id("10.0.0.1") == dv.stoi["10.0.0.1"]
    assert dv.to_id("missing") == dv.unk_index


def test_unpickle_old_module_path(tmp_path):
    """Old repo pickles reference model_architecture.vocab_builder.WordVocab.
    Simulate one and check load_vocab resolves it via the shim."""
    # Register parent package and child module in sys.modules
    model_arch_module = types.ModuleType("model_architecture")
    sys.modules["model_architecture"] = model_arch_module
    
    old_module_name = "model_architecture.vocab_builder"
    old_module = types.ModuleType(old_module_name)
    sys.modules[old_module_name] = old_module

    # Map the dummy class to the old module namespace
    DummyWordVocab.__module__ = old_module_name
    DummyWordVocab.__qualname__ = "WordVocab"
    old_module.WordVocab = DummyWordVocab

    v_old = DummyWordVocab(Counter({"a": 5, "b": 3, "c": 1}))
    p = tmp_path / "old_vocab.pkl"
    with open(p, "wb") as f:
        pickle.dump(v_old, f)

    # Clean up the modules from sys.modules to simulate them not being imported
    del sys.modules[old_module_name]
    del sys.modules["model_architecture"]

    loaded = WordVocab.load_vocab(str(p))
    # It gets unpickled as a WordVocab instance because _RenamingUnpickler maps
    # model_architecture.vocab_builder.WordVocab to logbert.vocab.WordVocab
    assert isinstance(loaded, WordVocab)
    assert loaded.itos == v_old.itos and loaded.stoi == v_old.stoi
