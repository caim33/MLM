class Fabric:
    def __init__(self, accelerator="auto", devices=1, **kwargs):
        try:
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            self.device = "cpu"

    def setup_module(self, module):
        try:
            module.device = self.device
        except Exception:
            pass
        return module
