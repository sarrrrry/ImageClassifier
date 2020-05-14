from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
from torch.utils.data import Dataset as torchDataset
from torchvision import datasets, transforms

from image_classifier.ml.device import Device


class FactoryDataLoader(metaclass=ABCMeta):
    def __init__(self, device: Device, batch_size: int, test_batch_size: Optional[int] = None):
        self.batch_size = batch_size
        if test_batch_size is None:
            self.test_batch_size = batch_size
        else:
            self.test_batch_size = test_batch_size

        self.kwargs = {'num_workers': 1, 'pin_memory': True} if device.is_cuda else {}

    @property
    @abstractmethod
    def train_dataset(self) -> torchDataset:
        pass

    @property
    @abstractmethod
    def valid_dataset(self) -> torchDataset:
        pass

    def create_train(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **self.kwargs
        )

    def create_test(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            **self.kwargs
        )


class DataLoader(FactoryDataLoader):
    def __init__(self, device: Device, batch_size: int, test_batch_size: Optional[int] = None):
        super(DataLoader, self).__init__(device, batch_size, test_batch_size)
        self.DATA_ROOT = "~/Data/PyTorch/"

    @property
    def train_dataset(self) -> torchDataset:
        return datasets.MNIST(str(self.DATA_ROOT), train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    @property
    def valid_dataset(self) -> torchDataset:
        return datasets.MNIST(str(self.DATA_ROOT), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))