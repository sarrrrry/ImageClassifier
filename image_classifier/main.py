import random
from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR

from image_classifier.configs.cfg_hyper_params import Cfg_HyperParams
from image_classifier.configs.cfg_optimizer import Cfg_Optimizer
from image_classifier.configs.cfg_train import Cfg_Train
from image_classifier.ml.data.dataloader import DataLoader
from image_classifier.ml.device import Device
from image_classifier.ml.networks.net import Net
from image_classifier.ml.optimizers.build import Optimizer
from image_classifier.ml.updater import ClassifierUpdater

logger = getLogger(__file__)

HYDRA_CFG_PATH = Path("../configs/ml/hydra_cfg.yaml")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from dataclasses import dataclass
from image_classifier.configs.base import BaseCfg


@dataclass
class Config(BaseCfg):
    hydra_cfg: DictConfig

    def __post_init__(self):
        self.train: Cfg_Train = Cfg_Train().assign_hydra(self.hydra_cfg.train)
        self.hyper_params: Cfg_HyperParams = Cfg_HyperParams().assign_hydra(self.hydra_cfg.hyper_params)
        self.optimizer: Cfg_Optimizer = Cfg_Optimizer().assign_hydra(self.hydra_cfg.optimizer)

    def logging(self):
        self.train.logging()
        self.hyper_params.logging()
        self.optimizer.logging()

    def asdict(self):
        return {
            "train": self.train.asdict(),
            "hyper_params": self.hyper_params.asdict(),
            "optimizer": self.optimizer.asdict(),
        }


class BaseMlWriter:
    def __init__(self):
        pass


import mlflow
from typing import Dict
from mlflow.tracking import MlflowClient

from collections.abc import MutableMapping


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class MlflowWriter(BaseMlWriter):
    def __init__(self, log_dir, experiment_name, **kwargs):
        super().__init__()

        mlflow_dir = log_dir / "mlflow" / "mlruns"
        self.client = MlflowClient(tracking_uri=str(mlflow_dir))
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def __del__(self):
        self.client.set_terminated(self.run_id)

    def log_params(self, params: Dict):
        # from flatten_dict import flatten
        flatten_params = flatten(params)

        for key, value in flatten_params.items():
            self.log_param(key, value)

    def log_artifact(self, local_path: Path):
        if local_path.exists():
            self.client.log_artifact(self.run_id, str(local_path))
        else:
            logger.info(f"NOT Exists: {local_path}")

    def log_torch_model(self, model):
        with mlflow.start_run(self.run_id):
            mlflow.pytorch.log_model(model, 'models')

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)


@hydra.main(config_path=HYDRA_CFG_PATH)
def main(hydra_cfg):
    # -------------------------
    # 設定の定義と出力
    # -------------------------
    cwd = Path.cwd()
    LOGS_ROOT = cwd.parents[2].resolve()
    experiment_name = cwd.parents[0].name  # depends on hydra config file

    logger.info(f"Current experiments name : \t'{experiment_name}'")
    logger.info(f"Logging to: \n\t{LOGS_ROOT}")

    cfg = Config(hydra_cfg)
    cfg.logging()

    # -------------------------
    # 準備
    # -------------------------
    set_seed(cfg.train.seed)
    device = Device()
    writer = MlflowWriter(LOGS_ROOT, experiment_name)
    writer.log_params(cfg.asdict())

    model = Net().to(device())
    optimizer = Optimizer(
        cfg.optimizer
    ).build(model.parameters())

    dataloader = DataLoader(
        device=device,
        batch_size=cfg.hyper_params.batch_size,
        test_batch_size=cfg.hyper_params.test_batch_size
    )

    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.hyper_params.step_gamma)
    updater = ClassifierUpdater(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dataloader=dataloader
    )

    try:
        updater.update(
            N_epochs=cfg.train.epochs,
            log_interval=cfg.train.log_interval
        )
    finally:
        if cfg.train.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")
        logger.info("=== all process is done ===")

        art_dir = cwd / ".hydra"
        writer.log_artifact(cwd / "main.log")
        writer.log_artifact(art_dir / "config.yaml")
        writer.log_artifact(art_dir / "hydra.yaml")
        writer.log_artifact(art_dir / "overrides.yaml")


if __name__ == '__main__':
    main()
