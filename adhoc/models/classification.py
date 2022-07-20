from .base import AcousticOdometryBase

import torchvision
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, TypeVar
from torchmetrics.functional import accuracy

T = TypeVar('T', bound='torch.nn.modules.Module')


class ClassificationBase(AcousticOdometryBase):
    cost_function = nn.CrossEntropyLoss()

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

    def get_Vx(self, prediction: torch.tensor) -> float:
        return self.centers[int(prediction.argmax(1).sum().item())]

    def get_label(self, sample: dict) -> torch.tensor:
        return torch.bucketize(sample['Vx'], boundaries=self.boundaries)


class UnnormalizedCNN(ClassificationBase):

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
        return F.log_softmax(x, dim=1)


class CNN(ClassificationBase):

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
        return F.log_softmax(x, dim=1)


class Resnet18(ClassificationBase):

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        lr: float = 0.0001,
        boundaries: np.ndarray = np.linspace(0.005, 0.065, 7)
        ):
        super().__init__(input_dim, lr, boundaries)
        # init a pretrained resnet
        backbone = torchvision.models.resnet18(pretrained=True)
        # Change input channel dimensions
        backbone.conv1 = nn.Conv2d(
            input_dim[0],
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
            )
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify
        self.classifier = nn.Linear(num_filters, self.output_dim)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x


class SqueezeNet(ClassificationBase):

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        lr: float = 0.0001,
        boundaries: np.ndarray = np.linspace(0.005, 0.065, 7)
        ):
        super().__init__(input_dim, lr, boundaries)
        # init a pretrained resnet
        self.backbone = torchvision.models.squeezenet1_1(pretrained=True)
        # Change input channel dimensions
        self.backbone.features[0] = nn.Conv2d(
            input_dim[0], 64, kernel_size=3, stride=2
            )
        # use the pretrained model to classify
        self.classifier = nn.Linear(1000, self.output_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


# TODO InformedBase


class UnnormalizedInformedCNN(AcousticOdometryBase):
    w_cost = nn.CrossEntropyLoss()
    s_cost = nn.CrossEntropyLoss()

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        lr: float = 0.0001,
        wheel_radius: float = 0.1,
        conv1_filters: int = 64,
        conv2_filters: int = 128,
        hidden_size: int = 512,
        ):
        super().__init__()
        self.wheel_radius = wheel_radius
        # Label is deg/s
        self.w_boundaries = torch.from_numpy(np.arange(2.5, 32.5, 5))
        # But prediction is rad/s
        self.w_centers = np.deg2rad(np.arange(0, 31, 5)).tolist()
        self.s_centers = [-0.3, 0.1, 0, 0.1, 0.3, 0.6, 1]
        self.s_boundaries = torch.from_numpy(
            np.array([-0.2, -0.05, 0.05, 0.2, 0.5, 0.8])
            )
        self.save_hyperparameters()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_dim[0], conv1_filters, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Conv2d(conv1_filters, conv2_filters, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            nn.Flatten(),
            )
        conv_output = torch.numel(
            self.feature_extractor(torch.zeros((1, *input_dim)))
            )
        self.w_classifier = nn.Sequential(
            nn.Linear(conv_output, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_size, len(self.w_centers)),
            nn.LogSoftmax(dim=1),
            )
        self.s_classifier = nn.Sequential(
            nn.Linear(conv_output, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden_size, len(self.s_centers)),
            nn.LogSoftmax(dim=1),
            )

    def forward(self, x):
        x = self.feature_extractor(x)
        w = self.w_classifier(x)
        s = self.s_classifier(x)
        return w, s

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        w, s = self(x.float())
        loss_w = self.w_cost(w, y[:, 0])
        loss_s = self.s_cost(s, y[:, 1])
        return {
            'loss': loss_w + loss_s,
            'loss_w': loss_w,
            'loss_s': loss_w,
            'acc_w': accuracy(w, y[:, 0]),
            'acc_s': accuracy(s, y[:, 1]),
            }

    def _odometry_step(
        self,
        batch,
        ):
        features = batch['features']
        frame_duration = batch['frame_duration']
        n_frames = features.shape[2]
        segment_frames = self.hparams['input_dim'][2]
        num_segments = int(n_frames - segment_frames)
        features = torch.from_numpy(features[np.newaxis, :, :, :]
                                    ).float().to(self.device)
        Vw = np.empty(num_segments)
        slip = np.empty(num_segments)
        for i in range(num_segments):
            prediction = self(features[:, :, :, i:i + segment_frames])
            Vw[i] = self.get_Vw(prediction)
            slip[i] = self.get_slip(prediction)
        start = batch['start_timestamp'
                      ] + frame_duration * (segment_frames - 1)
        timestamps = np.linspace(
            start,
            start + num_segments * frame_duration,
            num=num_segments,
            endpoint=True,
            )
        odom = pd.DataFrame({'Vw': Vw, 'slip': slip}, index=timestamps)
        odom['Vx'] = odom['Vw'] * self.wheel_radius * (1 - odom['slip'])
        skid = odom['slip'] < 0
        odom.loc[skid, 'Vx'] = (
            odom.loc[skid, 'Vw'] * self.wheel_radius /
            (1 + odom.loc[skid, 'slip'])
            )
        odom['tx'] = odom.index.to_series().diff() * odom['Vx']
        odom.iloc[0, :] = 0
        odom['X'] = odom['tx'].cumsum()
        return odom

    def get_Vw(self, prediction: torch.tensor) -> float:
        w = prediction[0]
        return self.w_centers[int(w.argmax(dim=1).sum().item())]

    def get_slip(self, prediction: torch.tensor) -> float:
        s = prediction[1]
        return self.s_centers[int(s.argmax(dim=1).sum().item())]

    def get_Vx(self, prediction: torch.tensor) -> float:
        w = self.get_Vw(prediction)
        s = self.get_slip(prediction)
        if s > 0:
            return w * self.wheel_radius * (1 - s)
        return w * self.wheel_radius / (1 + s)

    def get_label(self, sample: dict) -> torch.tensor:
        return torch.tensor((
            torch.bucketize(sample['Vw'], boundaries=self.w_boundaries),
            torch.bucketize(sample['slip'], boundaries=self.s_boundaries),
            ))
