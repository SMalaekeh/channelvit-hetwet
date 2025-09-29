import importlib

def test_import():
    m = importlib.import_module("channelvit_hetwet.train")
    assert hasattr(m, "LitChannelViT")
