from dataclasses import dataclass

from image_classifier.configs.base import BaseCfg


@dataclass
class Optim_Params(BaseCfg):
    lr: float = 1.0


@dataclass
class Cfg_Optimizer(BaseCfg):
    name: str = "Adadelta"
    lr: float = 1.0
