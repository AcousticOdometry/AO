import ao
import pytest

from pathlib import Path

@pytest.fixture(scope="session")
def output_folder():
    output_folder = Path(__file__).parent / "output"
    output_folder.mkdir(exist_ok=True, parents=True)
    return output_folder

@pytest.fixture(scope="session")
def data_folder():
    data_folder = Path(__file__).parent / "data"
    return data_folder

@pytest.fixture(scope="session")
def audio0(data_folder):
    return ao.io.wave_read(data_folder / 'audio0.wav')