def _lazy_load(path):
    import torch

    return torch.load(path, map_location="cpu")
