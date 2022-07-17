import ao
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pathlib import Path
from abc import abstractmethod


class AcousticOdometryBase(pl.LightningModule):

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        # return [optimizer], [lr_scheduler]
        return optimizer

    @abstractmethod
    def get_Vx(self, prediction: torch.tensor) -> float:
        pass

    @abstractmethod
    def get_label(self, sample: dict) -> torch.tensor:
        pass

    @abstractmethod
    def _shared_eval_step(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v
             for k, v in metrics.items() if 'loss' not in k},
            on_epoch=True,
            on_step=False,
            )
        self.log_dict(
            {f"train_{k}": v
             for k, v in metrics.items() if 'loss' in k},
            on_epoch=True,
            on_step=True,
            )
        return metrics['loss']

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
        Vx = np.empty(num_segments)
        for i in range(num_segments):
            prediction = self(features[:, :, :, i:i + segment_frames])
            Vx[i] = self.get_Vx(prediction)
        start = batch['start_timestamp'
                      ] + frame_duration * (segment_frames - 1)
        timestamps = np.linspace(
            start,
            start + num_segments * frame_duration,
            num=num_segments,
            endpoint=True,
            )
        Vx = pd.Series(Vx, index=timestamps)
        odom = pd.concat([Vx, Vx.index.to_series().diff() * Vx], axis=1)
        odom.columns = ['Vx', 'tx']
        odom.iloc[0, :] = 0
        odom['X'] = odom['tx'].cumsum()
        return odom

    def validation_step(
        self,
        batch,
        batch_idx,
        ):
        odom = self._odometry_step(batch)
        evaluation = ao.evaluate.odometry(
            odom, batch['ground_truth'], delta_seconds=1
            )
        for col in evaluation.columns:
            if 'RPE' in col:
                self.log(
                    'val_MRPE',
                    evaluation[col].mean(),
                    batch_size=1,
                    on_step=True,
                    on_epoch=True
                    )
            elif 'ATE' in col:
                self.log(
                    'val_MATE',
                    evaluation[col].mean(),
                    batch_size=1,
                    on_step=True,
                    on_epoch=True
                    )

    def test_step(
        self,
        batch,
        batch_idx,
        ):
        odom = self._odometry_step(batch)
        evaluation = ao.evaluate.odometry(
            odom, batch['ground_truth'], delta_seconds=1
            )
        if self.logger.log_dir:
            folder = Path(self.logger.log_dir) / batch['recording']
            folder.mkdir(exist_ok=True)
            odom.to_csv(
                folder / batch['file'].replace('.wav', '.odometry.csv'),
                index_label='timestamp',
                )
        for col in evaluation.columns:
            if 'RPE' in col:
                self.log(
                    'test_MRPE',
                    evaluation[col].mean(),
                    batch_size=1,
                    on_step=True,
                    on_epoch=True
                    )
            elif 'ATE' in col:
                self.log(
                    'test_MATE',
                    evaluation[col].mean(),
                    batch_size=1,
                    on_step=True,
                    on_epoch=True
                    )