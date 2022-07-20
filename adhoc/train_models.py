"""Train several Acoustic Odometry models

This script allows the user to train several Acoustic Odometry models. The
environment variables `MODELS_FOLDER` and `DATASETS_FOLDER` must be set and
will be used to determine the output location of the trained model and the
location of the training datasets, respectively.

The variable `models_to_train` is intended to be modified in order to determine
which model should be trained, with which dataset, and with which parameters.
"""
from train_model import train_model

import os
import numpy as np

from warnings import warn
from dotenv import load_dotenv

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser("Train multiple Acoustic Odometry models")
    parser.add_argument(
        '--datasets',
        '-d',
        default=None,
        type=str,
        nargs='+',
        )
    parser.add_argument(
        '--batch-size',
        '-b',
        default=32,
        type=int,
        )
    parser.add_argument(
        '--gpu',
        '-g',
        default=0,
        type=int,
        )
    args = parser.parse_args()
    if args.gpu < 0:
        raise ValueError(
            "GPU index must be non-negative, only one gpu is supported"
            )

    load_dotenv()
    models_folder = os.environ['MODELS_FOLDER']
    datasets_folder = os.environ['DATASETS_FOLDER']

    if args.gpu == 0:
        # Base
        # Smaller size 3
        # More bins 2
        models_to_train = {
            'base': {
                'dataset': 'base',
                'split_strategy': 'no-laptop',
                'task': 'classification',
                'boundaries': np.linspace(0.005, 0.065, 7),
                'architecture': 'UnnormalizedCNN',
                'conv1_filters': 64,
                'conv1_size': 5,
                'conv2_filters': 128,
                'conv2_size': 5,
                'hidden_size': 512,
                },
            'base-M': {
                'dataset': 'base',
                'split_strategy': 'no-laptop',
                'task': 'classification',
                'boundaries': np.linspace(0.005, 0.065, 7),
                'architecture': 'UnnormalizedCNN',
                'conv1_filters': 32,
                'conv1_size': 5,
                'conv2_filters': 64,
                'conv2_size': 5,
                'hidden_size': 256,
                },
            'base-S': {
                'dataset': 'base',
                'split_strategy': 'no-laptop',
                'task': 'classification',
                'boundaries': np.linspace(0.005, 0.065, 7),
                'architecture': 'UnnormalizedCNN',
                'conv1_filters': 16,
                'conv1_size': 5,
                'conv2_filters': 32,
                'conv2_size': 5,
                'hidden_size': 256,
                },
            'base-XS': {
                'dataset': 'base',
                'split_strategy': 'no-laptop',
                'task': 'classification',
                'boundaries': np.linspace(0.005, 0.065, 7),
                'architecture': 'UnnormalizedCNN',
                'conv1_filters': 8,
                'conv1_size': 5,
                'conv2_filters': 16,
                'conv2_size': 5,
                'hidden_size': 128,
                },
            'base-bins-14': {
                'dataset': 'base',
                'split_strategy': 'no-laptop',
                'task': 'classification',
                'boundaries': np.linspace(0.005, 0.065, 14),
                'architecture': 'UnnormalizedCNN',
                'conv1_filters': 64,
                'conv1_size': 5,
                'conv2_filters': 128,
                'conv2_size': 5,
                'hidden_size': 512,
                },
            'base-bins-28': {
                'dataset': 'base',
                'split_strategy': 'no-laptop',
                'task': 'classification',
                'boundaries': np.linspace(0.005, 0.065, 28),
                'architecture': 'UnnormalizedCNN',
                'conv1_filters': 64,
                'conv1_size': 5,
                'conv2_filters': 128,
                'conv2_size': 5,
                'hidden_size': 512,
                },
            }
        for seed in range(10, 13):
            models_to_train[f"normalized-files-{seed}"] = {
                'dataset': 'normalized-files',
                'split_strategy': 'no-laptop',
                'task': 'classification',
                'boundaries': np.linspace(0.005, 0.065, 7),
                'architecture': 'UnnormalizedCNN',
                'conv1_filters': 64,
                'conv1_size': 5,
                'conv2_filters': 128,
                'conv2_size': 5,
                'hidden_size': 512,
                'seed': seed
                }
        for seed in range(3, 6):
            models_to_train[f"normalized-files-batchnorm-{seed}"] = {
                'dataset': 'normalized-files',
                'split_strategy': 'no-laptop',
                'task': 'classification',
                'boundaries': np.linspace(0.005, 0.065, 7),
                'architecture': 'CNN',
                'conv1_filters': 64,
                'conv1_size': 5,
                'conv2_filters': 128,
                'conv2_size': 5,
                'hidden_size': 512,
                'seed': seed
                }
    elif args.gpu == 1:
        # Datasets 7 models
        models_to_train = {}
        for dataset in [
            'duration-5', 'duration-20', 'features-64', 'features-128',
            'segment-100', 'segment-140', 'double-channel'
            ]:
            models_to_train[dataset] = {
                'dataset': dataset,
                'split_strategy': 'no-laptop',
                'task': 'classification',
                'boundaries': np.linspace(0.005, 0.065, 7),
                'architecture': 'UnnormalizedCNN',
                'conv1_filters': 64,
                'conv1_size': 5,
                'conv2_filters': 128,
                'conv2_size': 5,
                'hidden_size': 512,
                }
    elif args.gpu == 2:
        # Split strategies and data agumentation 7
        models_to_train = {}
        for split_strategy in [
            'with-laptop', 'only-videomic', 'no-negative-slip', 'all-devices',
            'with-noise', 'with-gain', 'all-transforms'
            ]:
            models_to_train[split_strategy] = {
                'dataset': 'base',
                'split_strategy': split_strategy,
                'task': 'classification',
                'boundaries': np.linspace(0.005, 0.065, 7),
                'architecture': 'UnnormalizedCNN',
                'conv1_filters': 64,
                'conv1_size': 5,
                'conv2_filters': 128,
                'conv2_size': 5,
                'hidden_size': 512,
                }
    elif args.gpu == 3:
        # Ordinal classification 2
        # Regression 2
        # Unnormalized
        # Informed
        models_to_train = {}
        models_to_train['base-batchnorm'] = {
            'dataset': 'base',
            'split_strategy': 'no-laptop',
            'task': 'classification',
            'boundaries': np.linspace(0.005, 0.065, 7),
            'architecture': 'CNN',
            'conv1_filters': 64,
            'conv1_size': 5,
            'conv2_filters': 128,
            'conv2_size': 5,
            'hidden_size': 512,
            }
        for task, task_kwargs in zip(['regression', 'ordinal_classification'],
                                     [{}, {
                                         'boundaries':
                                             np.linspace(0.005, 0.065, 7)
                                         }]):
            for suffix, architecture in zip(['', '-batchnorm'],
                                            ['UnnormalizedCNN', 'CNN']):
                models_to_train[task.replace('_', '-') + suffix] = {
                    'dataset': 'base',
                    'split_strategy': 'no-laptop',
                    'task': task,
                    'architecture': architecture,
                    'conv1_filters': 64,
                    'conv1_size': 5,
                    'conv2_filters': 128,
                    'conv2_size': 5,
                    'hidden_size': 512,
                    **task_kwargs
                    }
        models_to_train['informed'] = {
            'dataset': 'base',
            'split_strategy': 'no-laptop',
            'task': 'classification',
            'architecture': 'UnnormalizedInformedCNN',
            }
    else:
        raise ValueError(f"Invalid gpu: {args.gpu}")
    for name, kwargs in models_to_train.items():
        if args.datasets and kwargs['dataset'] not in args.datasets:
            print(f"Skipping model {name} for dataset {kwargs['dataset']}")
            continue
        try:
            print(f"- Training model {name} with GPU {args.gpu}")
            trainer = train_model(
                name=name,
                models_folder=models_folder,
                batch_size=args.batch_size,
                gpus=[args.gpu],
                **{
                    'min_epochs': 20,
                    'max_epochs': 30,
                    'seed': 2,
                    **kwargs
                    }
                )
            if trainer.interrupted:
                warn(
                    f"Model {name} was interrupted. Skipping remaining models"
                    )
                break
        except ValueError as e:
            warn(f"Skip model {name}: {e}")