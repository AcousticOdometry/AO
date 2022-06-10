import numpy as np

from typing import List, Callable, Optional


def _frames(data: np.ndarray, length: int) -> List[np.ndarray]:
    """Splits the given audio signal in  non overlapping frames.

    Args:
        data (np.ndarray): Audio array with shape (n_samples, n_channels)
        length (int): Length of the frames in samples

    Returns:
        List[np.ndarray]: List of frames with shape (length, n_channels)
    """
    n_samples, _ = data.shape  # ! Will fail with a badly shaped data array
    num_frames = int(n_samples / length)
    return [data[n * length:(n + 1) * length, :] for n in range(num_frames)]


def frames(data: np.ndarray, sample_rate: int,
           duration: int) -> List[np.ndarray]:
    """Splits the given audio signal in  non overlapping frames.

    Args:
        data (np.ndarray): Audio array with shape (n_samples, n_channels)
        sample_rate (int): Frequency of samples in the audio signal [Hz]
        duration (int): Length of the frames in milliseconds

    Returns:
        List[np.ndarray]: List of frames with shape (length, n_channels)
    """
    return _frames(data, int(duration / 1000 * sample_rate))


def _segment(
    data: np.ndarray,
    length: int,
    overlap: int = 0,
    ) -> List[np.ndarray]:
    """Generates segments of the given audio signal.

    Args:
        data (np.ndarray): Audio array with shape (n_samples, n_channels)
        length (int): Length of the segments in samples
        overlap (int): Overlap between segments in samples

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
    # TODO accept lists of extractors
    extract: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
    """Extracts features from the given audio signal.

    Args:
        data (np.ndarray): Audio signal with shape (n_samples, n_channels)
        frame_samples (int): Number of samples in each frame
        extract (Callable[[np.ndarray], np.ndarray]): Function to extract
            features, it will be applied to each frame averaged in the channels
            axis. 

    Returns:
        np.ndarray: Array of features with shape 
            (int(n_samples / frame_samples), n_features) where n_features is
            determined by the extract function.
    """
    # TODO do not average channels
    f = np.vstack([
        extract(frame.mean(axis=1)) for frame in _frames(data, frame_samples)
        ]).transpose()
    return f


def segment_into_features(
    data: np.ndarray,
    sample_rate: int,
    frame_duration: int,
    segment_duration: Optional[int] = None,
    segment_overlap: int = 0,
    *,
    extract: Callable[[np.ndarray], np.ndarray],
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

    Returns:
        List[np.ndarray]: List of arrays of features
    """
    frame_samples = int(frame_duration * sample_rate / 1000)
    return [
        features(s, frame_samples, extract=extract)
        for s in segment(data, sample_rate, segment_duration, segment_overlap)
        ]
