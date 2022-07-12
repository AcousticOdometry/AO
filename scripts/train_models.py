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
        'base': {
            'dataset': 'base',
            'split_strategy': 'base',
            'architecture': 'CNN',
            'boundaries':  np.linspace(0.005, 0.065, 7),
            'conv1_filters': 64,
            'conv1_size': 5,
            'conv2_filters': 128,
            'conv2_size': 5,
            'hidden_size': 512,
            },
        'half-size': {
            'dataset': 'base',
            'split_strategy': 'base',
            'architecture': 'CNN',
            'boundaries':  np.linspace(0.005, 0.065, 7),
            'conv1_filters': 32,
            'conv1_size': 5,
            'conv2_filters': 64,
            'conv2_size': 5,
            'hidden_size': 256,
            },
        'segment-100': {
            'dataset': 'segment-100',
            'split_strategy': 'base',
            'architecture': 'CNN',
            'boundaries':  np.linspace(0.005, 0.065, 7),
            },
        'duration-5': {
            'dataset': 'duration-5',
            'split_strategy': 'base',
            'architecture': 'CNN',
            'boundaries':  np.linspace(0.005, 0.065, 7),
            },
        }
    for name, kwargs in models_to_train.items():
        try:
            train_model(
                name=name,
                models_folder=models_folder,
                batch_size=args.batch_size,
                gpus=args.gpus,
                min_epochs=10,
                max_epochs=100,
                **kwargs
                )
        except ValueError as e:
            warn(f"Skip model {name}: {e}")