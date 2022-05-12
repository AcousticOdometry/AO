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
    step = length - overlap
    num_segments = int((n_samples - overlap) / step)
    return [data[n * step:n * step + length, :] for n in range(num_segments)]


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


def features(
    data: np.ndarray,
    frame_samples: int,
    *,
    extract: Callable[[np.ndarray], np.ndarray],
    compression: Optional[Callable[[float], float]] = np.vectorize(math.log10),
    ) -> np.ndarray:
    """Extracts features from the given audio signal.

    Args:
        data (np.ndarray): Audio signal with shape (n_samples, n_channels)
        frame_samples (int): Number of samples in each frame
        extract (Callable[[np.ndarray], np.ndarray]): Function to extract
            features, it will be applied to each frame averaged in the channels
            axis. 
        compression (Optional[Callable[[float], float]], optional): Compression
            to be applied after the extract function. Defaults to 
            np.vectorize(math.log10).

    Returns:
        np.ndarray: Array of features with shape 
            (int(n_samples / frame_samples), n_features) where n_features is
            determined by the extract function.
    """
    f = np.vstack([
        extract(frame.mean(axis=1))
        for frame in _segment(data, frame_samples, 0)
        ]).transpose()
    # Compress features
    if compression:
        f = compression(f)
    return f


def segment_into_features(
    data: np.ndarray,
    sample_rate: int,
    frame_duration: int,
    segment_duration: Optional[int] = None,
    segment_overlap: int = 0,
    *,
    extract: Callable[[np.ndarray], np.ndarray],
    compression: Optional[Callable[[float], float]] = math.log10,
    ) -> List[np.ndarray]:
    """Segments the given audio signal and extracts features from each segment.

    Args:
        data (np.ndarray): Audio signal with shape (n_samples, n_channels)
        sample_rate (int): Frequency of samples in the audio signal [Hz]
        frame_duration (int): Length of a feature frame in milliseconds
        segment_duration (int): Length of a segment duration in milliseconds
        segment_overlap (int): Overlap between segments. Defaults to 0.
        extract (Callable[[np.ndarray], np.ndarray]): Function to extract
            features, it will be applied to each frame averaged in the channels
            axis. 
        compression (Optional[Callable[[float], float]], optional): Compression
            to be applied after the extract function. Defaults to 
            np.vectorize(math.log10).

    Returns:
        List[np.ndarray]: List of arrays of features
    """
    frame_samples = int(frame_duration * sample_rate / 1000)
    if compression:
        compression = np.vectorize(compression)
    return [
        features(s, frame_samples, extract=extract, compression=compression)
        for s in segment(data, sample_rate, segment_duration, segment_overlap)
        ]
