from ao.models.base import (
    RegressionBase, ClassificationBase, OrdinalClassificationBase
    )

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class RegressionCNN(RegressionBase):

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        output_dim: int,
        lr: float,
        conv1_filters: int = 64,
        conv1_size: int = 5,
        conv2_filters: int = 128,
        conv2_size: int = 5,
        hidden_size: int = 512,
        ):
        super().__init__(input_dim, output_dim, lr)
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
        fc1_input = torch.numel(self._forward(torch.zeros(input_dim)))
        self.fc1 = nn.Linear(fc1_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.hparams['output_dim'])

    def _forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        return self.flatten(x)

    def forward(self, x):
        x = self.normalization(x)
        x = self._forward(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.relu(self.fc2(x))


class ClassificationCNN(ClassificationBase):

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        output_dim: int,
        lr: float,
        conv1_filters: int = 64,
        conv1_size: int = 5,
        conv2_filters: int = 128,
        conv2_size: int = 5,
        hidden_size: int = 512,
        ):
        super().__init__(input_dim, output_dim, lr)
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
        fc1_input = torch.numel(self._forward(torch.zeros(input_dim)))
        self.fc1 = nn.Linear(fc1_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.hparams['output_dim'])

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
        return F.log_softmax(x, dim=1)


class OrdinalClassificationCNN(OrdinalClassificationBase):

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        output_dim: int,
        lr: float,
        conv1_filters: int = 64,
        conv1_size: int = 5,
        conv2_filters: int = 128,
        conv2_size: int = 5,
        hidden_size: int = 512,
        ):
        super().__init__(input_dim, output_dim, lr)
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
        fc1_input = torch.numel(self._forward(torch.zeros(input_dim)))
        self.fc1 = nn.Linear(fc1_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.hparams['output_dim'])

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