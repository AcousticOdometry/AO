from . import Extractor

import torch
import torchaudio
import numpy as np

from typing import Callable


class MFCC(Extractor):

    def __init__(
        self,
        num_samples: int = 1024,
        num_features: int = 64,
        sample_rate: int = 44100,
        on_channel: int = -1,
        ):
        super().__init__(
            num_samples, num_features, sample_rate, lambda x: x, on_channel
            )
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.num_features,
            log_mels=True,
            melkwargs={
                'n_fft': 1,
                'win_length': self.num_samples,
                'n_mels': self.num_features,
                },
            )

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        if self.on_channel < 0:
            samples = samples.mean(axis=0)
        else:
            samples = samples[self.on_channel, :]
        return self.mfcc(torch.from_numpy(samples).float()).numpy()