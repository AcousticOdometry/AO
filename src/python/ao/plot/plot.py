import ao
import math

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from typing import Tuple, Callable, Optional


def signal(
    data: np.ndarray,
    sample_rate: int,
    *,
    ax: plt.Axes = None,
    ) -> plt.Axes:
    num_samples, num_channels = data.shape  # ! This might fail if input is 3D
    if not ax:
        _, ax = plt.subplots()
    # Compute time vector
    time = np.linspace(0, num_samples / sample_rate, num_samples)
    for channel in range(num_channels):
        # Plot channel
        ax.plot(
            time, data[:, channel], linewidth=0.5, label=f"Channel {channel}"
            )
    # Format axes
    if num_channels > 1:
        ax.legend()
    ax.set_xlim((0, time[-1]))
    ax.set_xlabel("Time [s]")
    return ax


def features(
    data: np.ndarray,
    sample_rate: int,
    frame_samples: int,
    num_features: int,
    transform: Optional[Callable[[float], float]] = math.log10,
    *,
    extract: Optional[Callable[[np.array], np.array]] = None,
    extractor: ao.extractor.Extractor = ao.extractor.GammatoneFilterbank,
    ax: plt.Axes = None,
    pcolormesh_kwargs: dict = {},
    **extractor_kwargs,
    ) -> Tuple[QuadMesh, plt.Axes]:
    """Plot the features colormap of the given data.

    Args:
        data (np.ndarray): Input signal, shape (n_samples, n_channels).

        sample_rate (int): Samples per second of the input signal [Hz].
        
        frame_samples (int): Number of samples per frame.

        num_features (int): Number of features to extract per frame.

        extract (Callable(array-like) -> array-like, optional): Function that
            extracts features from a signal frame. If not provided it will be
            constructed using `extractor`. Defaults to None.

        extractor (ao.extractor.Extractor): Function factory for `extract`.
            Ignored if `extract` is provided. Defaults to
            ao.extractor.GammatoneFilterbank.

        transform (Callable(float) -> float, optional): Function to be
            applied to the extracted features. Defaults to `math.log10`.

        ax (plt.Axes, optional): Axes where to plot the gammatonegram. Defaults
            to None.

        pcolormesh_kwargs (dict): Keyword arguments for `pcolormesh`.

        **extractor_kwargs: Additional keyword arguments to pass to the
        `extractor`.

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
            transform=transform,
            **extractor_kwargs
            )
    # Extract features
    _features = ao.dataset.audio.features(
        data,
        frame_samples=frame_samples,
        extract=extract,
        )
    # Plot features
    if not ax:
        _, ax = plt.subplots()
    plot = ax.pcolormesh(_features, **pcolormesh_kwargs)
    # Add feature axis
    ax.set_yticks(np.linspace(0, num_features, 4))
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


def gammatonegram(
    data: np.ndarray,
    sample_rate: int,
    frame_samples: int,
    num_features: int,
    transform: Optional[Callable[[float], float]] = math.log10,
    *,
    low_Hz: Optional[int] = None,
    high_Hz: Optional[int] = None,
    temporal_integration: float = 0,
    ax: plt.Axes = None,
    pcolormesh_kwargs: dict = {},
    ) -> Tuple[QuadMesh, plt.Axes]:
    """Plot a gammatonegram of the given data.

    Args:
        data (np.ndarray): Input signal, shape (n_samples, n_channels).

        sample_rate (int): Samples per second of the input signal [Hz].
        
        frame_samples (int): Number of samples per frame.

        num_features (int): Number of gammatone filters to use.

        transform (Callable(float) -> float, optional): Function to be
            applied to the output of the gammatone filter. Defaults to
            `math.log10`.

        low_Hz (int, optional): Lowest center frequency to use in a filter.

        high_Hz (int, optional): Highest center frequency to use in a filter.

        temporal_integration (float, optional): Temporal integration in
        seconds.

        ax (plt.Axes, optional): Axes where to plot the gammatonegram. Defaults
        to None.

        pcolormesh_kwargs (dict): Keyword arguments for `pcolormesh`.

    Returns:
        Tuple[QuadMesh, plt.Axes]: Tuple containing the colormap object and the
        axes containing it.
    """
    kwargs = {'temporal_integration': temporal_integration}
    if low_Hz is not None:
        kwargs['low_Hz'] = low_Hz
    if high_Hz is not None:
        kwargs['high_Hz'] = high_Hz
    extract = ao.extractor.GammatoneFilterbank(
        num_samples=frame_samples,
        num_features=num_features,
        sample_rate=sample_rate,
        transform=transform,
        **kwargs
        )
    plot, ax = features(
        data=data,
        sample_rate=sample_rate,
        frame_samples=frame_samples,
        num_features=num_features,
        extract=extract,
        ax=ax,
        pcolormesh_kwargs=pcolormesh_kwargs,
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
