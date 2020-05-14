import torch


class Device:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def __call__(self) -> torch.device:
        return self.device

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"