from logging import getLogger

import torch
from torch.nn import functional as F

from image_classifier.ml.device import Device

logger = getLogger(__file__)


class ClassifierUpdater:
    def __init__(self, model, optimizer, scheduler, device: Device, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.torch_device = device()
        self.dataloader = dataloader

    def update(self, N_epochs=1, log_interval=1):
        for epoch in range(1, N_epochs + 1):
            self.train(log_interval, epoch)
            self.test()
            self.scheduler.step()

    def train(self, log_interval, epoch):
        self.model.train()
        train_loader = self.dataloader.create_train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.torch_device), target.to(self.torch_device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    def test(self):
        self.model.eval()
        test_loader = self.dataloader.create_test()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.torch_device), target.to(self.torch_device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
