import ao
import wave
import numpy as np

from pathlib import Path
from typing import Tuple
from functools import partial
from scipy.io.wavfile import write

import matplotlib.pyplot as plt


def _wave_read(path: Path) -> Tuple[np.ndarray, int]:
    """Reads content of a `wav` file into a numpy array.
    Args:
        file (Path): Path of the file to be opened and read.
    Returns:
        Tuple[np.ndarray, int]: Tuple containing the signal array and sampling
        rate.
    """
    with wave.open(str(path), mode='rb') as f:
        return (
            np.reshape(
                np.frombuffer(
                    f.readframes(f.getnframes()),
                    dtype=f'int{f.getsampwidth()*8}'
                    ),
                (-1, f.getnchannels()),
                ),
            f.getframerate(),
            )


def noise_snr(
    audio: np.ndarray,
    snr_linear: float,
    ) -> np.ndarray:
    # https://github.com/SuperKogito/pydiogment/blob/074543dc9483b450653f8a00c8279bf1eb873199/pydiogment/auga.py#L36
    audio = audio.copy()
    noise = np.random.randint(-100, 100, size=audio.shape, dtype=np.int16)
    # ao.plot.signal(noise, 44100)
    # plt.show()
    # compute powers
    noise_power = np.mean(np.power(noise, 2))
    audio_power = np.mean(np.power(audio, 2))
    # compute snr and scaling factor
    noise_factor = (audio_power / noise_power) * (1 / snr_linear)
    # add noise
    # with np.errstate(invalid='ignore'):
    audio += (np.sqrt(noise_factor) * noise).astype(np.int16)
    return audio


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(
        "Adds noise to an evaluation recording creating a different evaluation"
        )
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('snr', type=float)
    args = parser.parse_args()

    if args.input.is_dir():
        wav_files = sorted(args.input.glob('*.wav'))
    elif args.input.suffix == '.wav':
        wav_files = [args.input]
    else:
        raise ValueError(f"Invalid input: {args.input}")

    if not args.output.is_dir():
        raise ValueError(f"Output must be a directory not `{args.output}`")
    args.output.mkdir(parents=True, exist_ok=True)

    add_noise = partial(noise_snr, snr_linear=args.snr)

    for wav_file in wav_files:
        print(wav_file)
        audio, sample_rate = _wave_read(wav_file)
        noisy_audio = add_noise(audio)
        to = args.output / wav_file.name
        if to.exists():
            to.unlink()
        write(to, sample_rate, noisy_audio)