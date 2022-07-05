import librosa
import tempfile
import requests

import numpy as np

from pathlib import Path
from urllib.parse import urlparse
from typing import Tuple, Union


def _audio_read(
    path: Path,
    resample_to: int = None,
    mono: bool = False,
    **kwargs,
    ) -> Tuple[np.ndarray, int]:
    """Reads content of an audio file into a floating point numpy array.

    Args:
        path (Path): File path to be opened and read.
        resample_to (int, optional): If provided, audio will automatically be
            resampled to the given rate. Defaults to None.
        mono (bool, optional): If True, the signal will be converted to mono.
        kwargs (dict): Keyword arguments to pass to librosa.load.

    Returns:
        Tuple[np.ndarray, int]: Tuple containing the signal array and sampling
        rate.
    """
    data, sample_rate = librosa.load(
        path, sr=resample_to, mono=mono, dtype=np.float64, **kwargs
        )
    return np.atleast_2d(data), sample_rate


def audio_read(
    path: Union[str, Path],
    resample_to: int = None,
    mono: bool = False,
    **kwargs,
    ) -> Tuple[np.ndarray, int]:
    """Reads content of an audio file into a floating point numpy array.

    Args:
        path (Union[str, Path]): File string path, pathlib.Path or web URL to
            be downloaded. If it points to a file in the system, this will be
            opened and read. If it points to a web URL, it will be downloaded
            to a temporary file which will be opened and read.
        resample_to (int, optional): If provided, audio will automatically be
            resampled to the given rate. Defaults to None.
        mono (bool, optional): If True, the signal will be converted to mono.
        kwargs (dict): Keyword arguments to pass to librosa.load.

    Returns:
        Tuple[np.ndarray, int]: Tuple containing the signal array and sampling
        rate.
    """
    if Path(path).exists():  # If it is a file found in the system
        return _audio_read(path, resample_to=resample_to, mono=mono, **kwargs)
    str_path = str(path)
    urlparsed = urlparse(str_path)
    if all([urlparsed.scheme, urlparsed.netloc]):  # If it is a valid URL
        with tempfile.TemporaryFile() as temp:
            content = requests.get(str_path).content
            temp.write(content)
            temp.seek(0)
            return _audio_read(
                temp, resample_to=resample_to, mono=mono, **kwargs
                )
    raise ValueError(
        f'{str_path} is neither a URL nor a file found in the system'
        )
