"""Train a single Acoustic Odometry model

This script allows the user to train a single Acoustic Odometry model. This is
script is not intended to be modified and it should be used as a command line
tool. Provide the `--help` flag to see the available options.

This file can also be imported as a module in order to use the `train_model`
function.
"""
from multiprocessing.sharedctypes import Value
import ao
from ao.models import CNN

from gdrive import GDrive
from dataset import get_dataset_shards_and_config

import io
import os
import torch
import numpy as np
import webdataset as wds
import pytorch_lightning as pl

from pathlib import Path
from dotenv import load_dotenv
from typing import Callable, List, Dict, Tuple

LOCAL_FOLDER = Path(os.getcwd()) / 'models'
LOCAL_FOLDER.mkdir(parents=True, exist_ok=True)


def split_shards(shards: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    train = []
    validation = []
    for url, params in shards.items():
        if params['transform'] != 'None':
            continue
        elif params['device'] in [
            'rode-videomic-ntg-top', 'rode-smartlav-top'
            ]:
            validation.append(url)
        else:
            train.append(url)
    return train, validation


train_split_strategies = {'base': split_shards}


class WebDatasetModule(pl.LightningDataModule):

    def __init__(
        self,
        config: dict,
        train_shards: List[str],
        validation_shards: List[str] = [],
        batch_size: int = 6,
        # include_transforms: List[str] = [],
        ):
        super().__init__()
        self.config = config
        self.train_shards = train_shards
        self.validation_shards = validation_shards
        self.batch_size = batch_size
        # Applied to features
        self.get_batch_features = torch.from_numpy
        # TODO regression or classification
        # TODO Vx or slip + Vw ?
        # self.get_label = lambda result: torch.tensor(int(result['Vx'] * 100))
        # TODO self.dims
        # self.num_classes

    def get_batch_labels(self, batch):
        return torch.tensor([int(result['Vx']) * 100 for result in batch])

    def get_dataloader(self, shards):
        return wds.DataPipeline(
            wds.SimpleShardList(shards),
            wds.tarfile_to_samples(),
            wds.shuffle(5000),
            wds.decode(
                wds.handle_extension('.npy', lambda x: np.load(io.BytesIO(x)))
                ),
            wds.to_tuple('npy', 'json'),
            wds.batched(self.batch_size, partial=False),
            wds.map_tuple(self.get_batch_features, self.get_batch_labels),
            )

    def train_dataloader(self):
        return self.get_dataloader(shards=self.train_shards)

    def val_dataloader(self):
        return self.get_dataloader(shards=self.validation_shards)


def model_exists(name: str, models_folder: str) -> bool:
    folder_id = GDrive.get_folder_id(models_folder)
    if folder_id:
        gdrive = GDrive()
        # Look for a subfolder in `models_folder` with `name` as title
        for folder in gdrive.list_folder(folder_id):
            if not GDrive.is_folder(folder):
                continue
            elif folder['title'] == name:
                # Model exists if that subfolder contains a `model.pt` file
                for f in gdrive.list_folder(folder['id']):
                    if f['title'] == 'model.pt':
                        return True
                # If no `model.pt` file is found trash the subfolder
                folder.Trash()
            return False
    # Models folder is a local folder
    model_path = Path(models_folder) / name / 'model.pt'
    return model_path.exists()


def save_model(
    name: str,
    model: pl.LightningModule,
    models_folder: str,
    logging_dir: Path,
    ):
    # TODO Save also config
    folder_id = GDrive.get_folder_id(models_folder)
    if folder_id:
        gdrive = GDrive()
        model_folder = gdrive.create_folder(name, folder_id)
        model_folder.Upload()
        model_file = gdrive.create_file('model.pt', model_folder['id'])
        torch.jit.save(model.to_torchscript(), logging_dir / 'model.pt')
        model_file.SetContentFile(str(logging_dir / 'model.pt'))
        model_file.Upload()
    else:
        # Models folder is a local folder
        model_folder = Path(models_folder) / name
        model_folder.mkdir(exist_ok=True)
        torch.jit.save(model.to_torchscript(), model_folder / 'model.pt')


def train_model(
    name: str,
    dataset: str,
    split_shards: Callable[[Dict[str, dict]], Tuple[List[str], List[str]]],
    models_folder: str,
    batch_size: int = 32,
    max_epochs: int = 15,
    gpus: int = -1,
    ):
    # Check if model already exists
    if model_exists(name, models_folder):
        raise ValueError(f"Model {name} already exists in {models_folder}")
    # Get dataset
    shards, config = get_dataset_shards_and_config(dataset)
    train_shards, validation_shards = split_shards(shards)
    dataset = WebDatasetModule(
        config,
        train_shards=train_shards,
        validation_shards=validation_shards,
        batch_size=batch_size
        )
    # Initialize model
    # TODO use config
    model = CNN(classes=8)
    # Configure trainer and train
    logging_dir = LOCAL_FOLDER / name
    logging_dir.mkdir(exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=logging_dir)
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=max_epochs,
        logger=logger,
        default_root_dir=logging_dir,
        gpus=gpus,
        # auto_lr_find=True,
        # auto_scale_batch_size='binsearch',
        )
    trainer.tune(model, dataset)
    trainer.fit(model, dataset)
    # Save model
    return save_model(name, model, models_folder, logging_dir)


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
        type=str,
        default=os.getenv('MODELS_FOLDER'),
        help=(
            "Path to a directory where the model will be saved. A subfolder "
            "will be created with the `name` of the model, which will be "
            "considered the output folder."
            )
        )
    parser.add_argument(
        '--batch-size',
        '-b',
        default=32,
        type=int,
        )
    parser.add_argument(
        '--gpus',
        '-g',
        default=-1,
        type=int,
        nargs='+',
        )
    # TODO train_split
    args = parser.parse_args()

    # Parse output argument
    if not args.output:
        raise ValueError(
            "Missing output folder, provide --output argument or "
            "MODELS_FOLDER environmental variable"
            )

    train_model(
        name=args.name,
        dataset=args.dataset,
        split_shards=split_shards,
        models_folder=args.output,
        batch_size=args.batch_size,
        gpus=args.gpus,
        )
