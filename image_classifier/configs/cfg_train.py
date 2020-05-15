from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from image_classifier.configs.base import BaseCfg


@dataclass
class Cfg_Train(BaseCfg):
    """ 学習スクリプト用の設定変数
    """
    epochs: int = 1
    seed: int = 1
    no_cuda: bool = True
    log_interval: int = 10
    save_model: Optional[Path] = None
