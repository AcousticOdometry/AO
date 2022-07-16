"""Train a single Acoustic Odometry model

This script allows the user to train a single Acoustic Odometry model. This is
script is not intended to be modified and it should be used as a command line
tool. Provide the `--help` flag to see the available options.

This file can also be imported as a module in order to use the `train_model`
function.
"""
import ao

import models

from gdrive import GDrive
from wheel_test_bed_dataset import WheelTestBedDataset
from upload_model import upload_model, upload_odometry

import os
import torch
import shutil
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from tqdm import tqdm
from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from typing import List, Optional, Union, Callable
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

LOCAL_MODELS_FOLDER = Path(__file__).parent.parent / 'models'
LOCAL_MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

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
    evaluation_folder: Optional[str] = None,
    ):
    # Handle model first
    models_folder_id = GDrive.get_folder_id(models_folder)
    if models_folder_id:
        model_folder = Path(logger.log_dir)
    else:
        model_folder = Path(models_folder) / logger.name
        model_folder.mkdir(parents=True, exist_ok=True)
        # Copy from log_dir to models_folder
        for f in Path(logger.log_dir).iterdir():
            if f.is_file():
                shutil.copy(str(f), str(model_folder / f.name))
    # Save everything locally first
    torch.jit.save(model.to_torchscript(), model_folder / 'model.pt')
    ao.io.yaml_dump(dict(config), model_folder / 'model.yaml')
    ao.io.yaml_dump(dataset.config, model_folder / 'dataset.yaml')
    dataset.train_data.to_csv(
        model_folder / f"train_data.csv", index_label='index'
        )
    # Upload to Google Drive if needed
    if models_folder_id:
        upload_model(model_folder, models_folder_id)
    # Handle odometry data
    if not evaluation_folder:
        return
    evaluation_folder_id = GDrive.get_folder_id(evaluation_folder)
    if not evaluation_folder_id:
        raise NotImplementedError('Save odometry to local evaluation folder')
    upload_odometry(model_folder, evaluation_folder_id)


# Splitting

SPLIT_RNG = random.Random()


def _split_by_transform_and_devices(
    data: 'pd.DataFrame',
    config: dict,
    use_transforms: List[str] = ['None'],
    train_split: float = 0.8,
    test_devices: List[str] = ['rode-videomic-ntg-top', 'rode-smartlav-top'],
    val_devices: List[str] = [],
    filter_recordings: Callable[[dict], bool] = lambda params: True,
    ):
    use_recordings = [
        i for i, r in enumerate(config['recordings'])
        if filter_recordings(ao.dataset.parse_filename(r))
        ]
    train_indices, val_indices, test_indices = [], [], []
    for index, sample in data.iterrows():
        if sample['recording'] not in use_recordings:
            continue
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


SPLIT_STRATEGIES = {
    'train-with-laptop':
        partial(
            _split_by_transform_and_devices,
            use_transforms=['None'],
            train_split=1,
            test_devices=[],
            val_devices=['rode-videomic-ntg-top', 'rode-smartlav-top']
            ),
    'no-laptop':
        partial(
            _split_by_transform_and_devices,
            use_transforms=['None'],
            train_split=1,
            test_devices=['laptop-built-in-microphone'],
            val_devices=['rode-videomic-ntg-top', 'rode-smartlav-top']
            ),
    'only-videomic':
        partial(
            _split_by_transform_and_devices,
            use_transforms=['None'],
            train_split=1,
            test_devices=[
                'laptop-built-in-microphone', 'rode-smartlav-wheel-axis'
                ],
            val_devices=['rode-smartlav-top']
            ),
    'no-negative-slip':
        partial(
            _split_by_transform_and_devices,
            use_transforms=['None'],
            train_split=1,
            test_devices=['laptop-built-in-microphone'],
            val_devices=['rode-videomic-ntg-top', 'rode-smartlav-top'],
            filter_recordings=lambda r: any([
                r['s'] == 'nan',
                np.isnan(float(r['s'])),
                float(r['s']) >= 0,
                ]),
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
            train_split=1,
            test_devices=[],
            val_devices=['rode-videomic-ntg-top', 'rode-smartlav-top']
            ),
    'all-transforms':
        partial(
            _split_by_transform_and_devices,
            use_transforms=None,
            train_split=1,
            test_devices=[],
            val_devices=['rode-videomic-ntg-top', 'rode-smartlav-top']
            ),
    }

# Labeling


def _bucketize_sample(sample: dict, boundaries: np.ndarray, var: str):
    return torch.bucketize(
        sample[var],
        boundaries=boundaries,
        )


# Evaluation


def _odometry_step(
    batch,
    model: pl.LightningModule,
    get_Vx: callable,
    ):
    features = batch['features']
    frame_duration = batch['frame_duration']
    n_frames = features.shape[2]
    segment_frames = model.hparams['input_dim'][2]
    num_segments = int(n_frames - segment_frames)
    features = torch.from_numpy(features[np.newaxis, :, :, :]
                                ).float().to(model.device)
    Vx = np.empty(num_segments)
    for i in range(num_segments):
        prediction = model(features[:, :, :, i:i + segment_frames])
        Vx[i] = get_Vx(prediction)
    start = batch['start_timestamp'] + frame_duration * (segment_frames - 1)
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


def _validation_step(
    batch,
    batch_idx,
    model: pl.LightningModule,
    get_Vx: callable,
    ):
    odom = _odometry_step(batch, model, get_Vx)
    evaluation = ao.evaluate.odometry(
        odom, batch['ground_truth'], delta_seconds=1
        )
    for col in evaluation.columns:
        if 'RPE' in col:
            model.log(
                'val_MRPE',
                evaluation[col].mean(),
                batch_size=1,
                on_step=True,
                on_epoch=True
                )
        elif 'ATE' in col:
            model.log(
                'val_MATE',
                evaluation[col].mean(),
                batch_size=1,
                on_step=True,
                on_epoch=True
                )


def _test_step(
    batch,
    batch_idx,
    model: pl.LightningModule,
    get_Vx: callable,
    log_dir: Optional[Path] = None,
    ):
    odom = _odometry_step(batch, model, get_Vx)
    evaluation = ao.evaluate.odometry(
        odom, batch['ground_truth'], delta_seconds=1
        )
    if log_dir:
        folder = Path(log_dir) / batch['recording']
        folder.mkdir(exist_ok=True)
        odom.to_csv(
            folder / batch['file'].replace('.wav', '.odometry.csv'),
            index_label='timestamp',
            )
    for col in evaluation.columns:
        if 'RPE' in col:
            model.log(
                'test_MRPE',
                evaluation[col].mean(),
                batch_size=1,
                on_step=True,
                on_epoch=True
                )
        elif 'ATE' in col:
            model.log(
                'test_MATE',
                evaluation[col].mean(),
                batch_size=1,
                on_step=True,
                on_epoch=True
                )


def train_model(
    name: str,
    dataset: str,
    split_strategy: str,
    models_folder: str,
    architecture: str = 'CNN',
    task: str = 'Classification',
    task_options: dict = {
        'boundaries': np.linspace(0.005, 0.065, 7),
        },
    batch_size: int = 32,
    gpus: Union[int, List[int]] = -1,
    seed: Optional[int] = 1,
    min_epochs: int = 10,
    max_epochs: int = 20,
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
    if isinstance(seed, int):
        pl.seed_everything(seed, workers=True)
        dataset_rng = random.Random(seed)
        SPLIT_RNG.seed(seed)
    else:
        dataset_rng = None
    config['seed'] = seed
    # Get a model for the task and architecture
    model_class = getattr(ao.models, task + architecture)
    # TODO Should this be in dataset?
    config['task'] = task
    if task.endswith('Classification'):
        boundaries = task_options['boundaries']
        centers = list((boundaries[1:] + boundaries[:-1]) / 2)
        centers.insert(0, 2 * centers[0] - centers[1])
        centers.append(2 * centers[-1] - centers[-2])
        config['boundaries'] = boundaries.tolist()
        output_dim = len(boundaries) + 1
        get_label = partial(
            _bucketize_sample,
            boundaries=torch.from_numpy(boundaries),
            var='Vx',
            )
        if task.startswith('Ordinal'):
            get_Vx = lambda pred: centers[int(
                max(((pred > 0.5).cumprod(axis=1).sum(axis=1) - 1).item(), 0)
                )]
        else:
            get_Vx = lambda pred: centers[int(pred.argmax(1).sum().item())]
        learning_rate = 0.0001
    elif task == 'Regression':
        output_dim = 1
        # ! Ugly hardcoded normalization
        get_label = lambda sample: torch.tensor([sample['Vx'] * 10])
        get_Vx = lambda pred: (pred.item()) / 10
        learning_rate = 0.002
    else:
        raise NotImplementedError(f"{task = }")
    # Get dataset
    dataset = WheelTestBedDataset(
        dataset,
        split_data=SPLIT_STRATEGIES[split_strategy],
        batch_size=batch_size,
        get_label=get_label,
        shuffle=10E3,
        rng=dataset_rng,
        )
    print(f"Using dataset: {dataset.config['name']}")
    n_samples = len(dataset.train_data.index)
    print(
        f"train samples: {n_samples}, "
        f"batches: {int(n_samples / dataset.batch_size)}"
        )
    config['split_strategy'] = split_strategy
    # Initialize model
    model = model_class(
        input_dim=dataset.input_dim,
        output_dim=output_dim,
        lr=learning_rate,
        **model_kwargs
        )

    model.validation_step = partial(
        _validation_step, model=model, get_Vx=get_Vx
        )
    config['class'] = model_class.__name__
    # Configure trainer and train
    logger = pl.loggers.TensorBoardLogger(
        save_dir=LOCAL_MODELS_FOLDER, name=name
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        save_last=True
        # dirpath=logger.log_dir, save_top_k=2, monitor='val_MRPE', mode='min'
        )
    trainer = pl.Trainer(
        accelerator='auto',
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        logger=logger,
        gpus=gpus,
        num_sanity_val_steps=0,
        deterministic=True if seed else False,
        callbacks=[
            EarlyStopping(monitor='val_MRPE', mode='min'), checkpoint_callback
            ],
        )
    trainer.fit(model, dataset)
    # TODO this could be a separate script test_model.pt
    if checkpoint_callback.best_model_path:
        model = model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path
            )
    model.test_step = partial(
        _test_step, model=model, get_Vx=get_Vx, log_dir=Path(logger.log_dir)
        )
    if not trainer.interrupted:
        trainer.test(model, dataset)
    # Save model
    save_model(
        logger,
        model,
        config,
        dataset,
        models_folder,
        evaluation_folder=os.getenv('EVALUATION_FOLDER', None)
        )
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
