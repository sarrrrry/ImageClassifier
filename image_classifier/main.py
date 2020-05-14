import random
from logging import getLogger
from pathlib import Path

import hydra
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from image_classifier.ml.config import Config
from image_classifier.ml.data.dataloader import DataLoader
from image_classifier.ml.device import Device
from image_classifier.ml.networks.net import Net
from image_classifier.ml.updater import ClassifierUpdater

logger = getLogger(__file__)

HYDRA_CFG_PATH = Path("../configs/ml/config.yaml")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path=HYDRA_CFG_PATH)
def main(cfg):
    print(cfg)
    args = Config()
    set_seed(args.seed)

    device = Device()

    model = Net().to(device())
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    dataloader = DataLoader(
        device=device,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size
    )

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    updater = ClassifierUpdater(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dataloader=dataloader
    )
    # updater.update(N_epochs=args.epochs, log_interval=args.log_interval)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    logger.info("=== all process is done ===")


if __name__ == '__main__':
    main()
