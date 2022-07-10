import ao

import os
import pytest
import pandas as pd

from pathlib import Path
from warnings import warn
from dotenv import load_dotenv

load_dotenv()
models_folder = os.getenv('MODELS_FOLDER', None)
if models_folder:
    try:
        models_folder = Path(models_folder)
        model_folders = [f for f in models_folder.iterdir() if f.is_dir()]
    except Exception as e:
        warn(f"Could not load models folder: {e}")
        model_folders = []
else:
    model_folders = []


@pytest.fixture(scope='session', params=model_folders)
def model_folder(request):
    return request.param


@pytest.fixture(scope='session')
def output_folder():
    output_folder = Path(__file__).parent / 'output'
    output_folder.mkdir(exist_ok=True, parents=True)
    return output_folder


@pytest.fixture(scope='session')
def data_folder():
    data_folder = Path(__file__).parent / 'data'
    return data_folder


@pytest.fixture(
    scope='session',
    params=[
        'audio0.wav',
        r"https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/ratemap/t29_lwwj2n_m17_lgwe7s.wav",
        ],
    ids=['experiment0', 'mixed_speech']
    )
def audio_data(data_folder, request):
    url = request.param
    if not url.startswith('http'):
        url = data_folder / url
    return ao.io.audio_read(url)


@pytest.fixture(scope='session')
def odometry_estimations(data_folder) -> pd.DataFrame:
    return [
        pd.read_csv(f, index_col='timestamp')
        for f in sorted((data_folder / 'odometry').glob('*.odometry.csv'))
        ]


@pytest.fixture(scope='session')
def odometry_ground_truth(data_folder) -> pd.DataFrame:
    return pd.read_csv(
        data_folder / 'odometry' / 'ground_truth.csv', index_col='timestamp'
        )
