import wave
import tempfile
import requests

import numpy as np

from pathlib import Path
from urllib.parse import urlparse
from typing import BinaryIO, Tuple, Union


def _wave_read(file: Union[str, BinaryIO]) -> Tuple[np.ndarray, int]:
    """Reads content of a `wav` file into a numpy array.

    Args:
        file (Union[str, BinaryIO]): File path string or file-like object.

    Returns:
        Tuple[np.ndarray, int]: Tuple containing the signal array and sampling
        rate.
    """
    with wave.open(file, mode='rb') as f:
        return (
            np.reshape(
                np.frombuffer(
                    f.readframes(f.getnframes()),
                    dtype=f'int{f.getsampwidth()*8}'
                    ), (-1, f.getnchannels())
                ),
            f.getframerate(),
            )


def wave_read(path: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """Reads content of a `wav` file into a numpy array.

    Args:
        path (Union[str, Path]): File string path, pathlib.Path or web URL to
        be downloaded. If it points to a file in the system, this will be
        opened and read. If it points to a web URL, it will be downloaded to a
        temporary file which will be opened and read.

    Returns:
        Tuple[np.ndarray, int]: Tuple containing the signal array and sampling
        rate.
    """
    str_path = str(path)
    urlparsed = urlparse(str_path)
    if all([urlparsed.scheme, urlparsed.netloc]):  # If it is a valid URL
        with tempfile.TemporaryFile() as temp:
            content = requests.get(str_path).content
            temp.write(content)
            temp.seek(0)
            return _wave_read(temp)
    elif Path(path).exists():  # If it is a file found in the system
        return _wave_read(str_path)
    raise ValueError(
        f'{str_path} is neither a URL nor a file found in the system'
        )
