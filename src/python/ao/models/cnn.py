import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from torchmetrics.functional import accuracy


class CNN(pl.LightningModule):

    def __init__(
        self,
        input_dim: Tuple[int],
        # TODO output_dim: Tuple[int],
        classes: int,
        lr: float = 0.0001,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.cost_function = nn.CrossEntropyLoss()
        # TODO get configuration for sizes
        # TODO save configuration in config.yaml
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(171776, 512)
        self.fc2 = nn.Linear(512, self.hparams['classes'])

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr
            )
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [lr_scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        # print('Train!')
        prediction = self(x.float())
        # print(prediction)
        # print(y.long())
        loss = self.cost_function(prediction, y)
        acc = accuracy(prediction, y)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        prediction = self(x.float())
        loss = self.cost_function(prediction, y)
        acc = accuracy(prediction, y)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        prediction = self(x.float())
        loss = self.cost_function(prediction, y)
        acc = accuracy(prediction, y)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics