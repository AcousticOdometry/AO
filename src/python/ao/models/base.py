import pytorch_lightning as pl

import torch
import torch.nn as nn

from typing import Tuple

from torchmetrics.functional import accuracy


class ClassificationBase(pl.LightningModule):

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        output_dim: int,
        lr: float = 0.0001,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.cost_function = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [lr_scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizer = self.optimizers()
        optimizer.zero_grad()
        prediction = self(x.float())
        loss = self.cost_function(prediction, y)
        loss.backward()
        optimizer.step()
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        acc = accuracy(prediction, y)
        self.log('train_acc', acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x.float())
        loss = self.cost_function(prediction, y)
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        acc = accuracy(prediction, y)
        self.log('val_acc', acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x.float())
        loss = self.cost_function(prediction, y)
        self.log('test_loss', loss)
        acc = accuracy(prediction, y)
        self.log('test_acc', acc)
        return loss