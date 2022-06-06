import ao

import os
import pytest

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture(scope='session')
def output_folder():
    output_folder = Path(__file__).parent / 'output'
    output_folder.mkdir(exist_ok=True, parents=True)
    return output_folder


@pytest.fixture(scope='session')
def data_folder():
    data_folder = Path(__file__).parent / 'data'
    return data_folder

@pytest.fixture(scope='session')
def models_folder():
    models_folder = os.getenv('AO_MODELS_FOLDER', None)
    if not models_folder:
        pytest.skip("AO_MODELS_FOLDER not set")
    return models_folder


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
    return ao.io.wave_read(url)