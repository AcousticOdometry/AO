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

    models_to_train = {
        'regression': {
            'dataset': 'base',
            'split_strategy': 'no-laptop',
            'task': 'Regression',
            'architecture': 'CNN',
            'conv1_filters': 64,
            'conv1_size': 5,
            'conv2_filters': 128,
            'conv2_size': 5,
            'hidden_size': 512,
            },
        'regression-half-size': {
            'dataset': 'base',
            'split_strategy': 'no-laptop',
            'task': 'Regression',
            'architecture': 'CNN',
            'conv1_filters': 32,
            'conv1_size': 5,
            'conv2_filters': 64,
            'conv2_size': 5,
            'hidden_size': 256,
            },
        'regression-quarter-size': {
            'dataset': 'base',
            'split_strategy': 'no-laptop',
            'task': 'Regression',
            'architecture': 'CNN',
            'conv1_filters': 16,
            'conv1_size': 5,
            'conv2_filters': 32,
            'conv2_size': 5,
            'hidden_size': 256,
            },
        'regression-only-videomic': {
            'dataset': 'base',
            'split_strategy': 'only-videomic',
            'task': 'Regression',
            'architecture': 'CNN',
            'conv1_filters': 64,
            'conv1_size': 5,
            'conv2_filters': 128,
            'conv2_size': 5,
            'hidden_size': 512,
            },
        }
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
                min_epochs=10,
                max_epochs=20,
                **{
                    'seed': 1,
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