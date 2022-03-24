import ao
import math

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from typing import Tuple, Callable, Optional


def gammatonegram(
    data: np.ndarray,
    sample_rate: int,
    frame_samples: int,
    num_features: int,
    compression: Optional[Callable[[float], float]] = math.log10,
    *,
    low_Hz: Optional[int] = None,
    high_Hz: Optional[int] = None,
    ax: plt.Axes = None,
    ) -> Tuple[QuadMesh, plt.Axes]:
    """Plot a gammatonegram of the given data.

    Args:
        data (np.ndarray): Input signal, shape (n_samples, n_channels).

        sample_rate (int): Samples per second of the input signal [Hz].
        
        frame_samples (int): Number of samples per frame.

        num_features (int): Number of gammatone filters to use.

        compression (Callable(float) -> float, optional): Function to be
        applied to the output of the gammatone filter. Defaults to
        `math.log10`.

        low_Hz (int, optional): Lowest center frequency to use in a filter.

        high_Hz (int, optional): Highest center frequency to use in a filter.

        ax (plt.Axes, optional): Axes where to plot the gammatonegram. Defaults
        to None.

    Returns:
        Tuple[QuadMesh, plt.Axes]: Tuple containing the colormap object and the
        axes containing it.
    """
    kwargs = {}
    if low_Hz:
        kwargs['low_Hz'] = low_Hz
    if high_Hz:
        kwargs['high_Hz'] = high_Hz
    extract = ao.extractor.GammatoneFilterbank(
        num_samples=frame_samples,
        num_features=num_features,
        sample_rate=sample_rate,
        **kwargs
        )
    plot, ax = features(
        data=data,
        sample_rate=sample_rate,
        frame_samples=frame_samples,
        num_features=num_features,
        compression=compression,
        extract=extract,
        ax=ax,
        )
    # Change feature axis
    center_frequencies = [f.cf for f in extract.filters]
    yticks = []
    yticklabels = []
    for ytick in ax.get_yticks().astype(int).tolist():
        try:
            yticklabels.append(f"{center_frequencies[ytick]:.1E}")
            yticks.append(ytick)
        except IndexError:
            yticklabels.append(f"{center_frequencies[-1]:.1E}")
            yticks.append(len(center_frequencies))
            break
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("Center Frequency [Hz]")
    return plot, ax


def features(
        data: np.ndarray,
        sample_rate: int,
        frame_samples: int,
        num_features: int,
        compression: Optional[Callable[[float], float]] = math.log10,
        *,
        extract: Optional[Callable[[np.array], np.array]] = None,
        extractor: ao.extractor.Extractor = ao.extractor.GammatoneFilterbank,
        ax: plt.Axes = None,
        **kwargs
    ) -> Tuple[QuadMesh, plt.Axes]:
    """Plot the features colormap of the given data.

    Args:
        data (np.ndarray): Input signal, shape (n_samples, n_channels).

        sample_rate (int): Samples per second of the input signal [Hz].
        
        frame_samples (int): Number of samples per frame.

        num_features (int): Number of features to extract per frame.

        compression (Callable(float) -> float, optional): Function to be
        applied to the extracted features. Defaults to `math.log10`.

        extract (Callable(array-like) -> array-like, optional): Function that
        extracts features from a signal frame. If not provided it will be
        constructed using `extractor`. Defaults to None.

        extractor (ao.extractor.Extractor): Function factory for `extract`.
        Ignored if `extract` is provided. Defaults to
        ao.extractor.GammatoneFilterbank.

        ax (plt.Axes, optional): Axes where to plot the gammatonegram. Defaults
        to None.

        **kwargs: Additional keyword arguments to pass to the `extractor`.

    Returns:
        Tuple[QuadMesh, plt.Axes]: Tuple containing the colormap object and the
        axes containing it.
    """
    # Initialise the extract function
    if not extract:
        extract = extractor(
            num_samples=frame_samples,
            num_features=num_features,
            sample_rate=sample_rate,
            **kwargs
            )
    # Average signal accross channels
    data = data.copy().mean(axis=1)
    # Pad the data so it fits the gammatone filterbank
    num_frames = math.ceil(data.size / frame_samples)
    data = np.append(data, np.zeros(num_frames * frame_samples - data.size))
    # Extract features
    features = np.empty((num_features, num_frames))
    for frame in range(num_frames):
        batch = data[frame * frame_samples:(frame + 1) * frame_samples]
        batch_features = extract(batch)
        features[:, frame] = batch_features
    # Compress features
    if compression:
        features = np.vectorize(compression)(features)
    # Plot features
    if not ax:
        _, ax = plt.subplots()
    plot = ax.pcolormesh(
        np.flip(features, axis=0), # Flip rows, top should be latest feature
        cmap='jet',
        )
    # Add feature axis
    ax.set_ylabel("Features [-]")
    # Add time axis
    xlim = ax.get_xlim()
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([
        f"{x * frame_samples / sample_rate}" for x in ax.get_xticks()
        ])
    ax.set_xlim(xlim)
    ax.set_xlabel("Time [s]")
    return plot, ax