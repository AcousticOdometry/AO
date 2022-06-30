import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(pl.LightningModule):

    def __init__(self, classes: int):
        super().__init__()
        # TODO get configuration for sizes
        # TODO save configuration in config.yaml
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(171776, 512)
        self.fc2 = nn.Linear(512, classes)
        self.cost_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        prediction = self(x)
        loss = self.cost_function(
            prediction,
            # TODO remove hardcoded device. Lightning should take care
            torch.from_numpy(y).to('cuda').long()
            )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        prediction = self(x)
        test_loss = self.cost_function(
            prediction,
            # TODO remove hardcoded device. Lightning should take care
            torch.from_numpy(y).to('cuda').long()
            )
        self.log("test_loss", test_loss)