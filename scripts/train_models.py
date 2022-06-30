import ao
from ao.models import CNN

import io
import torch
import numpy as np
import webdataset as wds
import pytorch_lightning as pl

from pathlib import Path
from typing import Optional, List
from argparse import ArgumentParser


class WebDatasetModule(pl.LightningDataModule):

    def __init__(
        self,
        dataset_path: Path,
        batch_size: int = 6,
        exclude_devices: List[str] = [
            'rode-videomic-ntg-top', 'rode-smartlav-top'
            ],
        # include_transforms: List[str] = [],
        ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.exclude_devices = exclude_devices
        self.get_features = torch.from_numpy
        # TODO regression or classification
        # TODO Vx or slip + Vw ?
        self.get_label = lambda result: int(result['Vx'] * 100)
        # TODO self.dims
        # self.num_classes

    def is_test_shard(self, shard: Path):
        params = ao.dataset.parse_filename(shard.stem)
        if params['device'] in self.exclude_devices:
            return False
        # TODO remove hardcoded
        elif params['transform'] != 'None':
            return None
        return True

    def prepare_data(self):
        # ? download dataset from gdrive if dataset_path is a URL ?
        self.train_shards = []
        self.test_shards = []
        for shard in self.dataset_path.glob('*.tar'):
            is_test = self.is_test_shard(shard)
            if is_test is True:
                self.test_shards.append(shard)
            elif is_test is False:
                self.train_shards.append(shard)

    def get_dataloader(self, shards):
        return wds.DataPipeline(
            wds.SimpleShardList([f"file:{path}" for path in shards]),
            wds.tarfile_to_samples(),
            # wds.shuffle(5000),
            wds.decode(
                wds.handle_extension('.npy', lambda x: np.load(io.BytesIO(x)))
                ),
            wds.to_tuple('npy', 'json'),
            wds.map_tuple(self.get_features, self.get_label),
            wds.batched(self.batch_size, partial=False)
            )

    def train_dataloader(self):
        return self.get_dataloader(shards=self.train_shards)

    def test_dataloader(self):
        return self.get_dataloader(shards=self.test_shards)


def main(dataset_path: Path, output_path: Path):
    data = WebDatasetModule(dataset_path=dataset_path, )
    model = CNN(classes=6)
    logger = pl.loggers.TensorBoardLogger(save_dir=output_path)
    trainer = pl.Trainer(
        accelerator="gpu", logger=logger, default_root_dir=output_path
        )
    trainer.fit(model, data)


if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    # args = parser.parse_args()
    main(
        dataset_path=Path(
            r"G:\Shared drives\VAO\VAO_WheelTestBed-Experiment-2\datasets\test"
            ),
        output_path=Path(
            r"G:\Shared drives\VAO\VAO_WheelTestBed-Experiment-2\models"
            ),
        )
