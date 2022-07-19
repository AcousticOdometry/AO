from .base import AcousticOdometryBase

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from torchmetrics.functional import accuracy


class OrdinalClassificationBase(AcousticOdometryBase):
    cost_function = nn.MSELoss(reduction='none')

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        lr: float,
        boundaries: np.ndarray,
        ):
        super().__init__()
        self.boundaries = torch.from_numpy(boundaries)
        self.output_dim = len(boundaries) + 1
        self.centers = list((boundaries[1:] + boundaries[:-1]) / 2)
        self.centers.insert(0, 2 * self.centers[0] - self.centers[1])
        self.centers.append(2 * self.centers[-1] - self.centers[-2])
        self.save_hyperparameters()

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x.float())
        return {
            'loss': self.cost_function(prediction, y),
            'acc': accuracy(prediction, y)
            }

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

    def get_Vx(self, prediction: torch.tensor) -> float:
        return self.centers[int(self.decode_label(prediction))]

    def get_label(self, sample: dict) -> torch.tensor:
        return torch.bucketize(sample['Vx'], boundaries=self.boundaries)

    def ordinal_loss(self, prediction: torch.tensor, y: torch.tensor):
        # Create out encoded target with [batch_size, num_labels] shape
        encoded_y = torch.zeros_like(prediction)
        # Fill in ordinal target function, i.e. 1 -> [1,1,0,...]
        for i, label in enumerate(y):
            encoded_y[i, 0:label + 1] = 1
        return self.cost_function(prediction, encoded_y).sum(axis=1).mean()

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x.float())
        return {
            'loss': self.ordinal_loss(prediction, y),
            'acc': accuracy(self.decode_label(prediction), y)
            }


class UnnormalizedCNN(OrdinalClassificationBase):

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        lr: float = 0.0001,
        boundaries: np.ndarray = np.linspace(0.005, 0.065, 7),
        conv1_filters: int = 64,
        conv1_size: int = 5,
        conv2_filters: int = 128,
        conv2_size: int = 5,
        hidden_size: int = 512,
        ):
        super().__init__(input_dim, lr, boundaries)
        self.save_hyperparameters()
        self.conv1 = nn.Conv2d(
            input_dim[0], conv1_filters, kernel_size=conv1_size
            )
        self.conv2 = nn.Conv2d(
            conv1_filters, conv2_filters, kernel_size=conv2_size
            )
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        fc1_input = torch.numel(self._forward(torch.zeros((1, *input_dim))))
        self.fc1 = nn.Linear(fc1_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_dim)

    def _forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return self.flatten(x)

    def forward(self, x):
        x = self._forward(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return torch.sigmoid(x)


class CNN(OrdinalClassificationBase):

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        lr: float = 0.0001,
        boundaries: np.ndarray = np.linspace(0.005, 0.065, 7),
        conv1_filters: int = 64,
        conv1_size: int = 5,
        conv2_filters: int = 128,
        conv2_size: int = 5,
        hidden_size: int = 512,
        ):
        super().__init__(input_dim, lr, boundaries)
        self.save_hyperparameters()
        self.normalization = nn.BatchNorm2d(input_dim[0], affine=False)
        self.conv1 = nn.Conv2d(
            input_dim[0], conv1_filters, kernel_size=conv1_size
            )
        self.conv2 = nn.Conv2d(
            conv1_filters, conv2_filters, kernel_size=conv2_size
            )
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        fc1_input = torch.numel(self._forward(torch.zeros((1, *input_dim))))
        self.fc1 = nn.Linear(fc1_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.output_dim)

    def _forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return self.flatten(x)

    def forward(self, x):
        x = self.normalization(x)
        x = self._forward(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return torch.sigmoid(x)
