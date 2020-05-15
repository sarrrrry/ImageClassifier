from abc import ABCMeta
from dataclasses import dataclass, asdict
from logging import getLogger

import pandas as pd
from tabulate import tabulate

logger = getLogger(__file__)


@dataclass
class BaseCfg(metaclass=ABCMeta):
    def __str__(self):
        dict_ = self.asdict()
        df = pd.DataFrame(
            [str(v).lower() if isinstance(v, bool) else str(v) for v in dict_.values()],
            index=dict_.keys()
        )
        msg = f"\n{self.__class__.__name__}\n"
        msg += tabulate(df)
        msg += "\n"
        return msg

    def assign_hydra(self, hydra_cfg):
        for key, value in hydra_cfg.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.info(f"SKIP: {self.__class__.__name__} doesn't have 'key: {key}' as 'value: {value}'")
        return self

    def asdict(self):
        return asdict(self)

    def logging(self):
        logger.info(str(self))
