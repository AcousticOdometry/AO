import ao

import pytest
import numpy as np


class TestUtils:

    @pytest.mark.parametrize(
        "filename_and_result", [
            (
                "transform_None;device_laptop-built-in-microphone;shard_000",
                {
                    'transform': 'None',
                    'device': 'laptop-built-in-microphone',
                    'shard': 0
                    },
                ),
            (
                "w_30.00;s_-0.30;contact_true;load_5;duration_10;delay_3000;"
                "date_2022-06-24;time_19-18-08",
                {
                    'w': 30,
                    's': -0.3,
                    'contact': True,
                    'load': 5,
                    'duration': 10,
                    'delay': 3000,
                    'date': '2022-06-24',
                    'time': '19-18-08'
                    },
                ),
            ]
        )
    def test_parse_filename(self, filename_and_result):
        filename, expected_result = filename_and_result
        params = ao.dataset.utils.parse_filename(filename)
        assert params == expected_result
        # Check that the result is reconstructible
        _filename = ao.dataset.utils.dict_to_filename(params)
        assert params == ao.dataset.utils.parse_filename(_filename)

    def test_dict_to_filename(self):
        with pytest.raises(ValueError):
            ao.dataset.utils.dict_to_filename({'good-key': 'bad_value'})
        with pytest.raises(ValueError):
            ao.dataset.utils.dict_to_filename({'bad;key': 'good.value'})


class TestAudio:

    @pytest.mark.parametrize("segment_duration", [100, 200, 300, 500, 1000])
    @pytest.mark.parametrize(
        "overlap_ratio", [
            0.1,
            0.3,
            0.5,
            0.9,
            pytest.param(
                1,
                marks=pytest.mark.xfail(
                    strict=True, reason="Overlap is equal to segment duration"
                    )
                ),
            pytest.param(
                1.1,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Overlap is greater than segment duration"
                    )
                ),
            ]
        )
    def test_segment(self, audio_data, segment_duration, overlap_ratio):
        # Check that the overlapping between segments is correct
        segment_overlap = int(segment_duration * overlap_ratio)
        last_segment = None
        start = 0
        for segment in ao.dataset.audio.segment(
            *audio_data,
            duration=segment_duration,
            overlap=segment_overlap,
            ):
            end = start + segment_duration
            _, n_samples = segment.shape
            if last_segment is not None:
                np.testing.assert_equal(
                    segment[:, 0:int(n_samples * overlap_ratio)],
                    last_segment[:,
                                 int((1 / overlap_ratio - 1) * n_samples *
                                     overlap_ratio):n_samples]
                    )
            last_segment = segment.copy()
            start = end - segment_overlap
        # Assert that there are no segments remaining
        audio_duration = audio_data[0].shape[1] / audio_data[1]
        assert audio_duration - end < segment_duration

    @pytest.mark.parametrize("frame_duration", [100])
    @pytest.mark.parametrize("num_features", [256])
    @pytest.mark.parametrize("extractor", [ao.extractor.GammatoneFilterbank])
    def test_features(
            self, audio_data, frame_duration, num_features, extractor
        ):
        data, sample_rate = audio_data
        _, n_samples = data.shape
        frame_samples = int(frame_duration / 1000 * sample_rate)
        features = ao.dataset.audio.features(
            data,
            extractors=extractor(
                num_samples=frame_samples,
                num_features=num_features,
                sample_rate=sample_rate,
                )
            )
        assert features.shape == (
            1, num_features, int(n_samples / frame_samples)
            )
