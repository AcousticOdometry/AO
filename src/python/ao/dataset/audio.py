import ao
import numpy as np

from typing import List, Callable, Optional


def _frames(data: np.ndarray, length: int) -> List[np.ndarray]:
    """Splits the given audio signal in non overlapping frames.

    Args:
        data (np.ndarray): Audio array with shape (n_channels, n_samples)

        length (int): Length of the frames in samples

    Returns:
        List[np.ndarray]: List of frames with shape (n_channels, length)
    """
    _, n_samples = data.shape  # ! Will fail with a badly shaped data array
    num_frames = int(n_samples / length)
    return [data[:, n * length:(n + 1) * length] for n in range(num_frames)]


def frames(
    data: np.ndarray,
    sample_rate: int,
    duration: int,
    ) -> List[np.ndarray]:
    """Splits the given audio signal in non overlapping frames.

    Args:
        data (np.ndarray): Audio array with shape (n_channels, n_samples)

        sample_rate (int): Frequency of samples in the audio signal [Hz]

        duration (int): Length of the frames in milliseconds

    Returns:
        List[np.ndarray]: List of frames with shape (n_channels, length) with
            length being int(duration / 1000 * sample_rate)
    """
    return _frames(data, int(duration / 1000 * sample_rate))


def _segment(
    data: np.ndarray,
    length: int,
    overlap: int = 0,
    ) -> List[np.ndarray]:
    """Generates segments of the given audio signal.

    Args:
        data (np.ndarray): Audio array with shape (n_channels, n_samples)

        length (int): Length of the segments in samples

        overlap (int): Overlap between segments in samples

    Returns:
        List[np.ndarray]: List of segments with shape (n_channels, length) with
            length being int(duration / 1000 * sample_rate)
    """
    _, n_samples = data.shape  # ! Will fail with a badly shaped data array
    step = length - overlap
    num_segments = int((n_samples - overlap) / step)
    return [data[:, n * step:n * step + length] for n in range(num_segments)]


def segment(
    data: np.ndarray,
    sample_rate: int,
    duration: int,
    overlap: int,
    ) -> List[np.ndarray]:
    """Generates segments of the given audio signal. It performs some input
    validation as well as some basic conversions from milliseconds to samples.

    Args:
        data (np.ndarray): Audio array with shape (n_channels, n_samples)

        sample_rate (int): Frequency of samples in the audio signal [Hz]

        duration (int): Length of the segments in milliseconds 

        overlap (int): Overlap between segments in milliseconds

    Raises:
        TypeError: Overlap should not be larger or equal than duration

    Returns:
        List[np.ndarray]: List of segments with shape (n_channels, length) with
            length being int(duration / 1000 * sample_rate)
    """
    if overlap >= duration:
        raise TypeError(
            f"Overlap between segments {overlap} must be smaller than the "
            f"segment duration {duration}"
            )
    segment_samples = int(duration * sample_rate / 1000)
    overlap_samples = int(overlap * sample_rate / 1000)
    return _segment(data, segment_samples, overlap_samples)


def features(
    data: np.ndarray,
    extractors: List[ao.extractor.Extractor],
    ) -> np.ndarray:
    """Extracts features from the given audio signal.

    Args:
        data (np.ndarray): Audio signal with shape (n_channels, n_samples)

        extractors (List[ao.extractor.Extractor]): Feature extractor, it will
            be applied to each frame.

    Returns:
        np.ndarray: Array of features with shape 
            (len(extractors), n_features, int(n_samples / frame_samples)) where
            n_features and frame_samples is determined by the extractors.
    """
    if not isinstance(extractors, list):
        extractors = [extractors]
    # Check all extractors are compatible
    n_channels, n_samples = data.shape
    n_features = extractors[0].num_features
    frame_samples = extractors[0].num_samples
    for extractor in extractors:
        if n_features != extractor.num_features:
            raise ValueError(
                "All extractors must have the same number of output features, "
                f"{n_features} != {extractor.num_features}"
                )
        if frame_samples != extractor.num_samples:
            raise ValueError(
                "All extractors must have the same number of frame samples, "
                f"{frame_samples} != {extractor.num_samples}"
                )
        if extractor.on_channel >= n_channels:
            raise ValueError(
                f"Extractor {extractor} is applied to a channel not present "
                f"in the data: {extractor.on_channel} >= {n_channels}"
                )
    # Preallocate array for features
    num_frames = int(n_samples / frame_samples)
    features = np.empty((len(extractors), n_features, num_frames))
    # * Harcoded from `_frames` to avoid recomputing number of frames
    frames = [
        data[:, n * frame_samples:(n + 1) * frame_samples]
        for n in range(num_frames)
        ]
    for i, extractor in enumerate(extractors):
        for j, frame in enumerate(frames):
            features[i, :, j] = extractor(frame)
    return features
    # TODO benchmark against no preallocation
    # return np.stack([
    #     np.column_stack([
    #         extract(frame) for frame in frames
    #         ]) for extract in extractors
    #     ])