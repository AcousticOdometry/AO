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
from pathlib import Path
from dotenv import load_dotenv


if __name__ == '__main__':
    load_dotenv()
    models_folder = os.environ['MODELS_FOLDER']
    datasets_folder = os.environ['DATASETS_FOLDER']

    models_to_train = {
        'base_cnn': {'dataset': 'base'}
    }
    for name, kwargs in models_to_train.items():
        train_model(
            dataset_folder=datasets_folder / kwargs.pop('dataset'),
            output_folder=models_folder / name,
            **kwargs
            )