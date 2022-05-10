import math
import numpy as np

from typing import List, Callable, Optional


def _segment(
    data: np.ndarray,
    length: int,
    overlap: int,
    ) -> List[np.ndarray]:
    """Generates segments of the given audio signal.

    Args:
        data (np.ndarray): Audio array with shape (n_samples, n_channels)
        length (int): Length of the segments in samples
        overlap (int): Overlap between sements  in samples

    Returns:
        List[np.ndarray]: List of segments with shape (length, n_channels)
    """
    n_samples, _ = data.shape  # ! Will fail with a badly shaped data array
    num_segments = int(n_samples / length)
    return [
        data[n * (length - overlap):n * (length - overlap) + length, :]
        for n in range(num_segments)
        ]


def segment(
    data: np.ndarray, 
    sample_rate: int,
    duration: int,
    overlap: int,
    ) -> List[np.ndarray]:
    """Generates segments of the given audio signal. It performs some input
    validation as well as some basic conversions from milliseconds to samples.

    Args:
        data (np.ndarray): Audio array with shape (n_samples, n_channels)
        sample_rate (int): Frequency of samples in the audio signal [Hz]
        duration (int): Length of the segments in milliseconds
        overlap (int): Overlap between segments in milliseconds

    Raises:
        TypeError: Overlap should not be larger or equal than duration

    Returns:
        List[np.ndarray]: List of segments with shape (length, n_channels)
    """
    if overlap >= duration:
        raise TypeError(
            f"Overlap between segments {overlap} must be smaller than the "
            f"segment duration {duration}"
            )
    # TODO check data array
    segment_samples = int(duration * sample_rate / 1000)
    overlap_samples = int(overlap * sample_rate / 1000)
    return _segment(data, segment_samples, overlap_samples)


def _features(
    data: np.ndarray,  # shape: (n_samples, n_channels)
    frame_samples: int,  # [samples]
    *,
    extract: Callable,
    compression: Optional[Callable[[float], float]] = math.log10,
    ):
    features = np.vstack([
        extract(frame.mean(axis=1))
        for frame in _segment(data, frame_samples, 0)
        ]).transpose()
    # Compress features
    if compression:
        features = np.vectorize(compression)(features)
    return features


def features():
    pass