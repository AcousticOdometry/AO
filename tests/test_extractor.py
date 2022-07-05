import ao

import math
import pytest
import warnings
import numpy as np

# TODO test override Extractor


@pytest.mark.parametrize('frame_duration', [1, 10, 100, 1000])
@pytest.mark.parametrize('num_features', [64, 256, 0])
def test_extractor(audio_data, frame_duration, num_features):
    data, fs = audio_data
    num_samples = int(frame_duration / 1000 * fs)  # samples per frame
    extractor = ao.extractor.GammatoneFilterbank(num_samples, num_features, fs)
    for frame in ao.dataset.audio._frames(data, num_samples):
        output = extractor(frame)
        assert len(output) == num_features


@pytest.mark.parametrize('frame_duration', [10, 100])
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
        raw = no_transform(frame)
        transformed = transform_log10(frame)
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


@pytest.mark.parametrize('frame_duration', [10, 100])
@pytest.mark.parametrize('num_features', [64, 256])
@pytest.mark.parametrize('on_channel', [-1, 0, 1, 2, 3])
def test_on_channel(audio_data, frame_duration, num_features, on_channel):
    data, fs = audio_data
    num_channels, _ = data.shape
    if on_channel >= num_channels:
        pytest.skip(f"Channel {on_channel} is not available")
    num_samples = int(frame_duration / 1000 * fs)  # samples per frame
    extractor = ao.extractor.GammatoneFilterbank(
        num_samples, num_features, fs, on_channel=on_channel
        )
    for k, frame in enumerate(ao.dataset.audio._frames(data, num_samples)):
        # Process frame outside of the extractor
        f = frame[on_channel, :] if on_channel >= 0 else np.mean(frame, axis=0)
        np.testing.assert_equal(extractor(frame), extractor(f.tolist()))
        if k > 20:
            break