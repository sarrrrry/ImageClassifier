from dataclasses import dataclass

from image_classifier.configs.base import BaseCfg


@dataclass
class Cfg_HyperParams(BaseCfg):
    step_gamma: float = 0.7
    batch_size: int = 64
    test_batch_size: int = 1000