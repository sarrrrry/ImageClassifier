from dataclasses import dataclass


@dataclass
class Config:
    # Training settings
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 1
    lr: float = 1.0
    gamma: float = 0.7
    no_cuda: bool = True
    seed: int = 1
    log_interval: int = 10
    save_model: bool = True
