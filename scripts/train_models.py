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

from warnings import warn
from dotenv import load_dotenv

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser("Train multiple Acoustic Odometry models")
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

    load_dotenv()
    models_folder = os.environ['MODELS_FOLDER']
    datasets_folder = os.environ['DATASETS_FOLDER']

    models_to_train = {
        'equalized-base-augmented': {
            'dataset': 'equalized-base-augmented',
            'validation_split': 0.2,
            'shard_selection_strategy': 'base',
            },
        'wtb-experiment-1-v2': {
            'dataset': 'wtb-experiment-1',
            'validation_split': 0.2,
            'shard_selection_strategy': 'base',
            },
        }
    for name, kwargs in models_to_train.items():
        try:
            train_model(
                name=name,
                models_folder=models_folder,
                batch_size=args.batch_size,
                gpus=args.gpus,
                **kwargs
                )
        except ValueError as e:
            warn(f"Skip model {name}: {e}")