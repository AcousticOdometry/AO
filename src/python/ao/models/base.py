import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from abc import abstractmethod
from torchmetrics.functional import accuracy


class AcousticOdometryBase(pl.LightningModule):

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        output_dim: int,
        lr: float,
        ):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [lr_scheduler]
        return optimizer

    @abstractmethod
    def _shared_eval_step(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v
             for k, v in metrics.items() if k != 'loss'},
            on_epoch=True,
            )
        self.log('train_loss', metrics['loss'], on_epoch=True, on_step=True)
        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        self.log_dict(
            {f"val_{k}": v
             for k, v in metrics.items()},
            on_epoch=True,
            )
        return metrics['loss']

    def test_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        self.log_dict(
            {f"test_{k}": v
             for k, v in metrics.items()},
            on_epoch=True,
            )
        return metrics['loss']


class ClassificationBase(AcousticOdometryBase):
    cost_function = nn.CrossEntropyLoss()

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x.float())
        return {
            'loss': self.cost_function(prediction, y),
            'acc': accuracy(prediction, y)
            }


class OrdinalClassificationBase(AcousticOdometryBase):
    cost_function = nn.MSELoss(reduction='none')

    def ordinal_loss(self, prediction: torch.tensor, y: torch.tensor):
        # Create out encoded target with [batch_size, num_labels] shape
        encoded_y = torch.zeros_like(prediction)
        # Fill in ordinal target function, i.e. 1 -> [1,1,0,...]
        for i, label in enumerate(y):
            encoded_y[i, 0:label + 1] = 1
        return self.cost_function(prediction, encoded_y).sum(axis=1).mean()

    @staticmethod
    def decode_label(prediction: torch.tensor):
        """Convert ordinal predictions to class labels, e.g.
        
        [0.9, 0.1, 0.1, 0.1] -> 0
        [0.9, 0.9, 0.1, 0.1] -> 1
        [0.9, 0.9, 0.9, 0.1] -> 2
        etc.
        """
        label = (prediction > 0.5).cumprod(axis=1).sum(axis=1) - 1
        F.threshold(label, 0, 0, inplace=True)
        return label

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x.float())
        return {
            'loss': self.ordinal_loss(prediction, y),
            'acc': accuracy(self.decode_label(prediction), y)
            }