import ao

import wave
import math
import tempfile
import requests
import numpy as np

from typing import Tuple


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





if __name__ == "__main__":
    audio_url = "https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/ratemap/t29_lwwj2n_m17_lgwe7s.wav"
    data, fs = wave_read(audio_url)
    frame_length = 10  # [ms]
    num_samples = int(np.ceil(frame_length / 1000 * fs))  # samples per frame
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(data)
    gammatonegram, _ = plot_gammatonegram(data, fs, num_samples, 64, ax=axs[1])
    fig.colorbar(gammatonegram, orientation='horizontal')
    plt.show()
