"""Train a single Acoustic Odometry model

This script allows the user to train a single Acoustic Odometry model. This is
script is not intended to be modified and it should be used as a command line
tool. Provide the `--help` flag to see the available options.

This file can also be imported as a module in order to use the `train_model`
function.
"""
import ao
from ao.models import CNN

from dataset import get_dataset_shards_and_config
from gdrive import GDrive

import io
import os
import torch
import numpy as np
import webdataset as wds
import pytorch_lightning as pl

from pathlib import Path
from warnings import warn
from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple

def split_shards(shards: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    train = []
    test = []
    for url, params in shards.items():
        if params['transform'] != 'None':
            continue
        elif params['device'] in ['rode-videomic-ntg-top', 'rode-smartlav-top']
            test.append(url)
        else:
            train.append(url)
    return train, test

train_split_strategies = {
    'base': split_shards
}


class WebDatasetModule(pl.LightningDataModule):

    def __init__(
        self,
        train_shards: List[Path],
        test_shards: List[Path] = [],
        val_shards: List[Path] = [],
        batch_size: int = 6,
        exclude_devices: List[str] = [
            'rode-videomic-ntg-top', 'rode-smartlav-top'
            ],
        # include_transforms: List[str] = [],
        ):
        super().__init__()
        self.dataset_folder = dataset_folder
        # Get dataset config
        self.dataset_config = {}
        for config_file in dataset_folder.glob('*.yaml'):
            file_params = ao.dataset.parse_filename(config_file.stem)
            key = next(iter(file_params))
            self.dataset_config.setdefault(key, {}).update({
                file_params[key]: ao.io.yaml_load(config_file)
                })
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
        # ? download dataset from gdrive if dataset_folder is a URL ?
        self.train_shards = []
        self.test_shards = []
        for shard in self.dataset_folder.glob('*.tar'):
            is_test = self.is_test_shard(shard)
            if is_test is True:
                self.test_shards.append(shard)
            elif is_test is False:
                self.train_shards.append(shard)

    def get_dataloader(self, shards):
        return wds.DataPipeline(
            wds.SimpleShardList(shards),
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


def save_model(model: pl.LightningModule, to: str):
    raise NotImplementedError


def train_model(
    name: str,
    dataset: str,
    split_shards: Callable[[Dict[str, dict]], Tuple[List[str], List[str]]],
    output_folder: Path,
    batch_size: int = 32,
    max_epochs: int = 15,
    ):
    shards, config = get_dataset_shards_and_config(dataset)
    test_shards, train_shards = split_shards(shards)
    # TODO Filter shards
    # TODO local_output_folder
    if output_folder.exists():
        warn(f"{output_folder} already exists, model will be overwritten")
    output_folder.mkdir(parents=True, exist_ok=True)
    dataset = WebDatasetModule(dataset_folder=dataset_folder)
    model = CNN(classes=6)
    logger = pl.loggers.TensorBoardLogger(save_dir=output_folder)
    trainer = pl.Trainer(
        max_epochs=max_epochs, logger=logger, default_root_dir=output_folder
        )
    trainer.fit(model, dataset)
    torch.jit.save(model.to_torchscript(), output_folder / "model.pt")


if __name__ == '__main__':
    from argparse import ArgumentParser

    load_dotenv()
    parser = ArgumentParser("Train a single Acoustic Odometry model")
    parser.add_argument(
        'name',
        type=str,
        help=(
            "Name of the model to be trained. Together with `--output` will "
            "determine the output folder."
            )
        )
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        required=True,
        # TODO gdrive is also accepted
        help=(
            "Path to the dataset. If DATASETS_FOLDER environment variable is "
            "set, the path provided here can be relative to that folder."
            )
        )
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        default=os.getenv('MODELS_FOLDER'),
        help=(
            "Path to a directory where the model will be saved. A subfolder "
            "will be created with the `name` of the model, which will be "
            "considered the output folder."
            )
        )
    args = parser.parse_args()

    # Parse output argument
    if args.output is None:
        raise ValueError(
            "Missing output folder, provide --output argument or "
            "MODELS_FOLDER environmental variable"
            )
    output_folder = args.output / args.name

    train_model(
        name=args.name,
        dataset=args.dataset,
        # TODO this should be an argument
        split_shards=split_shards,
        output_folder=output_folder,
        )
