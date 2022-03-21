import ao

import wave
import math
import tempfile
import requests
import numpy as np

from typing import Tuple
from matplotlib import pyplot as plt


def wave_read(url: str) -> Tuple[np.ndarray, int]:
    # Utility function that reads the whole `wav` file content into a numpy
    with tempfile.TemporaryFile() as temp:
        content = requests.get(url).content
        temp.write(content)
        temp.seek(0)
        with wave.open(temp, mode='rb') as f:
            return (
                np.reshape(
                    np.frombuffer(
                        f.readframes(f.getnframes()),
                        dtype=f'int{f.getsampwidth()*8}'
                        ), (-1, f.getnchannels())
                    ),
                f.getframerate(),
                )


def test_extractor():
    audio_url = "https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/ratemap/t29_lwwj2n_m17_lgwe7s.wav"
    data, fs = wave_read(audio_url)
    frame_length = 10  # [ms]
    num_samples = int(np.ceil(frame_length / 1000 * fs))  # samples per frame
    num_frames = math.ceil(data.size / num_samples)
    padded_data = np.append(
        data, np.zeros(num_frames * num_samples - data.size)
        )
    extractor = ao.GammatoneFilterbank(num_samples, 64, fs, 50, 8000)
    for index in range(num_frames):
        batch = padded_data[index * num_samples:(index + 1) * num_samples]
        output = extractor(batch)
        print([math.log10(x) for x in output])
