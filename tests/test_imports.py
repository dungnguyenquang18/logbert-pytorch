import importlib

def test_package_imports():
    for mod in ["logbert", "logbert.data", "logbert.model", "logbert.training"]:
        assert importlib.import_module(mod) is not None
