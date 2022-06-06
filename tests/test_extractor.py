import ao

import math
import pytest
import warnings
import numpy as np

# TODO test overide Extractor


@pytest.mark.parametrize('frame_duration', [1, 10, 100, 1000])
@pytest.mark.parametrize('num_features', [64, 256, 0])
def test_extractor(audio_data, frame_duration, num_features):
    data, fs = audio_data
    num_samples = int(frame_duration / 1000 * fs)  # samples per frame
    extractor = ao.extractor.GammatoneFilterbank(num_samples, num_features, fs)
    for frame in ao.dataset.audio.frames(data, num_samples):
        # TODO do not average channels, use extractor on_channel
        output = extractor(frame.mean(axis=1))
        assert len(output) == num_features


@pytest.mark.parametrize('frame_duration', [100])
@pytest.mark.parametrize('num_features', [256])
def test_transform(audio_data, frame_duration, num_features):
    data, fs = audio_data
    num_samples = int(frame_duration / 1000 * fs)  # samples per frame
    no_transform = ao.extractor.GammatoneFilterbank(
        num_samples, num_features, fs, lambda x: x
        )
    transform_log10 = ao.extractor.GammatoneFilterbank(
        num_samples, num_features, fs, lambda x: math.log10(x)
        )
    for frame in ao.dataset.audio.frames(data, fs, frame_duration):
        raw = no_transform(frame.mean(axis=1))
        transformed = transform_log10(frame.mean(axis=1))
        assert all([r != t for r, t in zip(raw, transformed)])
        assert all(np.log10(raw) == transformed)


def test_gammatone_filterbank():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        extractor = ao.extractor.GammatoneFilterbank()
        for filter in extractor.filters:
            assert isinstance(filter.cf, float)
            assert isinstance(filter.gain, float)
            assert isinstance(filter.a, list)
