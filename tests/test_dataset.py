from ao.dataset import audio

import math
import pytest
import numpy as np

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
        for segment in audio.segment(
            *audio_data,
            duration=segment_duration,
            overlap=segment_overlap,
            ):
            end = start + segment_duration
            n_samples, _ = segment.shape
            if last_segment is not None:
                np.testing.assert_equal(
                    segment[0:int(n_samples * overlap_ratio), :],
                    last_segment[int((1 / overlap_ratio - 1) * n_samples *
                                     overlap_ratio):n_samples, :]
                    )
            last_segment = segment.copy()
            start = end - segment_overlap
        # Assert that there are no segments remaining
        audio_duration = audio_data[0].shape[0] / audio_data[1]
        assert audio_duration - end < segment_duration
        

    # TODO def test_features(self, audio_data):