from logging import getLogger
from pathlib import Path
from typing import MutableMapping, Dict

import mlflow
from mlflow.tracking import MlflowClient

logger = getLogger(__file__)


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class BaseMlWriter:
    def __init__(self):
        pass


class MlflowWriter(BaseMlWriter):
    def __init__(self, log_dir, experiment_name):
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
