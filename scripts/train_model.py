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
import shutil
import random
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from typing import List, Optional, Union
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

CACHE_FOLDER = Path(__file__).parent.parent / 'models'
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

# Utils


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
    logger: pl.loggers.TensorBoardLogger,
    model: pl.LightningModule,
    config: dict,
    dataset: pl.LightningDataModule,
    models_folder: str,
    ):
    folder_id = GDrive.get_folder_id(models_folder)
    # Save everything locally first
    if folder_id:
        model_folder = Path(logger.log_dir)
    else:
        model_folder = Path(models_folder) / logger.name
        model_folder.mkdir(parents=True, exist_ok=True)
        # Copy from log_dir to models_folder
        for f in Path(logger.log_dir).iterdir():
            shutil.copy(str(f), str(model_folder / f.name))
    torch.jit.save(model.to_torchscript(), model_folder / 'model.pt')
    ao.io.yaml_dump(dict(config), model_folder / 'model.yaml')
    ao.io.yaml_dump(dataset.config, model_folder / 'dataset.yaml')
    for split in ['train', 'val', 'test']:
        getattr(dataset, f"{split}_data").to_csv(
            model_folder / f"{split}_data.csv", index_label='index'
            )
    # Upload to Google Drive if needed
    if folder_id:
        # Create model folder
        gdrive = GDrive()
        upload_to = gdrive.create_folder(logger.name, folder_id)
        upload_to.Upload()
        # Upload all files
        for f in tqdm([f for f in model_folder.iterdir() if f.is_file()],
                      desc='Upload model',
                      unit='file'):
            gdrive_file = gdrive.create_file(f.name, upload_to['id'])
            gdrive_file.SetContentFile(str(f))
            gdrive_file.Upload()


# Splitting

SPLIT_RNG = random.Random()


def _split_by_transform_and_devices(
    data: 'pd.DataFrame',
    config: dict,
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
        elif SPLIT_RNG.uniform(0, 1) <= train_split:
            train_indices.append(index)
        else:
            val_indices.append(index)
    return (train_indices, val_indices, test_indices)


# TODO _split_no_negative_slip

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
    'with-noise':
        partial(
            _split_by_transform_and_devices,
            use_transforms=['None', 'add-random-snr-noise'],
            train_split=0.8,
            test_devices=['rode-videomic-ntg-top', 'rode-smartlav-top'],
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

# Labeling


def _bucketize_sample(sample: dict, boundaries: np.ndarray, var: str):
    return torch.bucketize(
        sample[var],
        boundaries=boundaries,
        )


def train_model(
    name: str,
    dataset: str,
    split_strategy: str,
    models_folder: str,
    architecture: str = 'CNN',
    boundaries: Optional[np.ndarray] = np.linspace(0.005, 0.075, 8),
    batch_size: int = 32,
    gpus: Union[int, List[int]] = -1,
    seed: Optional[int] = 42,
    min_epochs: int = 10,
    max_epochs: int = 50,
    **model_kwargs,
    ) -> pl.Trainer:
    # Check if model already exists
    if model_exists(name, models_folder):
        raise ValueError(f"Model {name} already exists in {models_folder}")
    # Initialize model
    config = {
        'task': None,
        'split_strategy': split_strategy,
        'architecture': architecture
        }
    if seed:
        pl.seed_everything(seed, workers=True)
        dataset_rng = random.Random(seed)
        SPLIT_RNG.seed(seed)
        config['seed'] = seed
    else:
        dataset_rng = None
        config['seed'] = seed
    if boundaries is not None:
        config['task'] = 'classification'
        config['boundaries'] = boundaries.tolist()
        output_dim = len(boundaries) + 1
        get_label = partial(
            _bucketize_sample,
            boundaries=torch.from_numpy(boundaries),
            var='Vx',
            )
    else:
        raise ValueError(
            'Could not determine whether to use classification or regression'
            )
    # Get dataset
    dataset = WheelTestBedDataset(
        dataset,
        split_data=SPLIT_STRATEGIES[split_strategy],
        batch_size=batch_size,
        get_label=get_label,
        rng=dataset_rng,
        )
    print(f"Using dataset: {dataset.config['name']}")
    for split in ['train', 'val', 'test']:
        n_samples = len(getattr(dataset, f'{split}_data').index)
        print(
            f"{split} samples: {n_samples}, "
            f"batches: {int(n_samples / dataset.batch_size)}"
            )
    config['split_strategy'] = split_strategy
    # Initialize model
    if architecture == 'CNN':
        model_class = ao.models.CNN
        learning_rate = 0.0001
    else:
        raise ValueError(f"Unknown architecture {architecture}")
    model = model_class(
        input_dim=dataset.input_dim,
        output_dim=output_dim,
        lr=learning_rate,
        **model_kwargs
        )
    config['class'] = model_class.__name__
    # Configure trainer and train
    logger = pl.loggers.TensorBoardLogger(save_dir=CACHE_FOLDER, name=name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir, save_top_k=2, monitor="val_acc"
        )
    trainer = pl.Trainer(
        accelerator='auto',
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger,
        gpus=gpus,
        deterministic=True if seed else False,
        callbacks=[
            # EarlyStopping(monitor='val_acc', mode='max', patience=10),
            checkpoint_callback
            ],
        )
    trainer.fit(model, dataset)
    if checkpoint_callback.best_model_path:
        model = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path
            )
    if not trainer.interrupted:
        trainer.test(model, dataset)
    # Save model
    save_model(logger, model, config, dataset, models_folder)
    return trainer


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
        split_strategy=args.split_strategy,
        models_folder=args.output,
        batch_size=args.batch_size,
        gpus=args.gpus,
        )
