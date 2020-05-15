from dataclasses import dataclass

from torch import optim

from image_classifier.configs.base import BaseCfg


class Optimizer:
    @dataclass
    class defaults:
        name: str = "Adadelta"

    optimizers = {
        "Adadelta": optim.Adadelta
    }

    def __init__(self, cfg: BaseCfg):
        # name = cfg.name
        self.cfg = cfg

    def build(self, model_params):
        cfg_as_dict = self.cfg.asdict()

        name = cfg_as_dict.pop("name", self.defaults.name)
        optim = self.optimizers[name]

        return optim(model_params, **cfg_as_dict)