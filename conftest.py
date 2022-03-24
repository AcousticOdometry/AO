import pytest

from pathlib import Path

@pytest.fixture(scope="session")
def output_folder():
    output_folder = Path(__file__).parent / "tests" / "output"
    output_folder.mkdir(exist_ok=True, parents=True)
    return output_folder