import ao
import math

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import QuadMesh
from typing import List, Tuple, Dict, Callable, Optional


def signal(
    data: np.ndarray,
    sample_rate: int,
    *,
    ax: plt.Axes = None,
    ) -> plt.Axes:
    num_channels, num_samples = data.shape  # ! This might fail if input is 3D
    if not ax:
        _, ax = plt.subplots()
    # Compute time vector
    time = np.linspace(0, num_samples / sample_rate, num_samples)
    for channel in range(num_channels):
        # Plot channel
        ax.plot(
            time, data[channel, :], linewidth=0.5, label=f"Channel {channel}"
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
    extractors: List[ao.extractor.Extractor],
    *,
    axs: List[plt.Axes] = None,
    pcolormesh_kwargs: dict = {},
    **extractor_kwargs,
    ) -> Tuple[QuadMesh, plt.Axes]:
    """Plot the features colormap of the given data.

    Args:
        data (np.ndarray): Input signal, shape (n_samples, n_channels).

        sample_rate (int): Samples per second of the input signal [Hz].
        
        extractors (List[ao.extractor.Extractor]): Feature extractor, it will
            be applied to each frame.

        ax (plt.Axes, optional): Axes where to plot the gammatonegram. Defaults
            to None.

        pcolormesh_kwargs (dict): Keyword arguments for `pcolormesh`.

        **extractor_kwargs: Additional keyword arguments to pass to the
        `extractor`.

    Returns:
        Tuple[QuadMesh, plt.Axes]: Tuple containing the colormap object and the
        axes containing it.
    """
    if not isinstance(extractors, list):
        extractors = [extractors]
    # Extract features
    _features = ao.dataset.audio.features(data, extractors=extractors)
    # Plot features
    if not axs:
        _, axs = plt.subplots(len(extractors), 1, sharex=True, squeeze=False)
        axs = axs.flatten()
    for i, ax in enumerate(axs):
        plot = ax.pcolormesh(_features[i, :, :], **pcolormesh_kwargs)
        # Add feature axis
        ax.set_yticks(np.linspace(0, extractors[0].num_features, 4))
        ax.set_ylabel("Features [-]")
        # Add time axis
        xlim = ax.get_xlim()
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([
            f"{x * extractors[0].num_samples / sample_rate}"
            for x in ax.get_xticks()
            ])
        ax.set_xlim(xlim)
        ax.set_xlabel("Time [s]")
    return plot, axs


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
    extractor = ao.extractor.GammatoneFilterbank(
        num_samples=frame_samples,
        num_features=num_features,
        sample_rate=sample_rate,
        transform=transform,
        **kwargs
        )
    plot, (ax, *_) = features(
        data=data,
        sample_rate=sample_rate,
        extractors=[extractor],
        axs=[ax],
        pcolormesh_kwargs=pcolormesh_kwargs,
        )
    # Change feature axis
    center_frequencies = [f.cf for f in extractor.filters]
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