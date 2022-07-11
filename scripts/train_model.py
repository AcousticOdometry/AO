"""Train a single Acoustic Odometry model

This script allows the user to train a single Acoustic Odometry model. This is
script is not intended to be modified and it should be used as a command line
tool. Provide the `--help` flag to see the available options.

This file can also be imported as a module in order to use the `train_model`
function.
"""
import ao

from gdrive import GDrive
from wheel_test_bed_dataset import WheelTestBedDataset

import os
import torch
import random
import pytorch_lightning as pl

from tqdm import tqdm
from typing import List
from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from pytorch_lightning.callbacks import EarlyStopping

CACHE_FOLDER = Path(__file__).parent.parent / 'models'
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


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
    dataset: pl.LightningDataModule,
    models_folder: str,
    ):
    folder_id = GDrive.get_folder_id(models_folder)
    # Save everything locally first
    if folder_id:
        model_folder = CACHE_FOLDER / name
    else:
        model_folder = Path(models_folder) / name
    torch.jit.save(model.to_torchscript(), model_folder / 'model.pt')
    ao.io.yaml_dump(model.hparams, model_folder / 'hparams.yaml')
    ao.io.yaml_dump(dataset.config, model_folder / 'dataset.yaml')
    for split in ['train', 'val', 'test']:
        getattr(dataset, f"{split}_data").to_csv(
            model_folder / f"{split}_data.csv", index_label='index'
            )
    # Upload to Google Drive if needed
    if folder_id:
        # Create model folder
        gdrive = GDrive()
        upload_to = gdrive.create_folder(name, folder_id)
        upload_to.Upload()
        # Upload all files
        for f in tqdm([f for f in model_folder.iterdir() if f.is_file()],
                      desc='Upload model',
                      unit='file'):
            gdrive_file = gdrive.create_file(f.name, upload_to['id'])
            gdrive_file.SetContentFile(str(f))
            gdrive_file.Upload()


def _split_by_transform_and_devices(
    data,
    use_transforms: List[str] = ['None'],
    train_split: float = 0.8,
    test_devices: List[str] = ['rode-videomic-ntg-top', 'rode-smartlav-top'],
    val_devices: List[str] = [],
    ):
    train_indices, val_indices, test_indices = [], [], []
    for index, sample in data.iterrows():
        if use_transforms is not None:
            if sample['transform'] not in use_transforms:
                continue
        if sample['device'] in test_devices:
            test_indices.append(index)
        elif sample['device'] in val_devices:
            val_indices.append(index)
        elif random.uniform(0, 1) < train_split:
            train_indices.append(index)
        else:
            val_indices.append(index)
    return (train_indices, val_indices, test_indices)


SPLIT_STRATEGIES = {
    'base':
        partial(
            _split_by_transform_and_devices,
            use_transforms=['None'],
            train_split=0.8,
            test_devices=['rode-videomic-ntg-top', 'rode-smartlav-top'],
            val_devices=[],
            ),
    'validate-other-devices':
        partial(
            _split_by_transform_and_devices,
            use_transforms=['None'],
            train_split=1,
            test_devices=[],
            val_devices=['rode-videomic-ntg-top', 'rode-smartlav-top']
            ),
    'all-devices':
        partial(
            _split_by_transform_and_devices,
            use_transforms=['None'],
            train_split=0.8,
            test_devices=[],
            val_devices=[],
            ),
    'all-transforms':
        partial(
            _split_by_transform_and_devices,
            use_transforms=None,
            train_split=0.8,
            test_devices=['rode-videomic-ntg-top', 'rode-smartlav-top'],
            val_devices=[],
            ),
    }


def train_model(
        name: str,
        dataset: str,
        split_strategy: str,
        models_folder: str,
        batch_size: int = 32,
        gpus: int = -1,
        min_epochs: int = 10,
        max_epochs: int = 20,
        **kwargs
    ):
    # Check if model already exists
    if model_exists(name, models_folder):
        raise ValueError(f"Model {name} already exists in {models_folder}")
    # Get dataset
    dataset = WheelTestBedDataset(
        dataset,
        split_data=SPLIT_STRATEGIES[split_strategy],
        batch_size=batch_size,
        get_label=lambda result: torch.tensor(round(result['Vx'] * 100)),
        )
    dataset.config['split_strategy'] = split_strategy
    # Initialize model
    # TODO use dataset for output_dim
    model = ao.models.CNN(input_dim=dataset.input_dim, output_dim=7, **kwargs)
    # Configure trainer and train
    logging_dir = CACHE_FOLDER / name
    logging_dir.mkdir(exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=logging_dir)
    trainer = pl.Trainer(
        accelerator='auto',
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger,
        default_root_dir=logging_dir,
        gpus=gpus,
        callbacks=[EarlyStopping(monitor='val_acc', mode='max')]
        # auto_lr_find=True,
        # auto_scale_batch_size='binsearch',
        )
    # trainer.tune(model, dataset)
    trainer.fit(model, dataset)
    trainer.test(model, dataset)
    # Save model
    return save_model(name, model, dataset, models_folder)


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
        '--split-strategy',
        '-s',
        default=list(SPLIT_STRATEGIES.keys())[0],
        type=str,
        choices=SPLIT_STRATEGIES.keys()
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
    # TODO split_strategy
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
        split_strategy='base',
        models_folder=args.output,
        batch_size=args.batch_size,
        gpus=args.gpus,
        )
