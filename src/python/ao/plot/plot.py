import ao

import math
import numpy as np

from matplotlib import pyplot as plt

def gammatonegram(data, fs, frame_length, num_filters=64, *, ax=None):
    extractor = ao.GammatoneFilterbank(frame_length, num_filters, fs, 50, 8000)
    num_frames = math.ceil(data.size / num_samples)
    padded_data = np.append(
        data, np.zeros(num_frames * num_samples - data.size)
        )
    features = []
    for index in range(num_frames):
        batch = padded_data[index * num_samples:(index + 1) * num_samples]
        output = extractor(batch)
        # features.append(output)
        features.append([math.log10(x) for x in output])
    # Plot the gammatonegram
    if not ax:
        _, ax = plt.subplots()
    features = list(map(list, zip(*features)))  # Transpose
    gammatonegram = ax.pcolormesh(features)
    return gammatonegram, ax


def features(
        data: np.ndarray,
        sample_rate: int,
        frame_samples: int,
        num_features: int,
        *,
        ax: plt.Axes = None
    ):
    pass