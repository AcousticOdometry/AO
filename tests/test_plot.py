import ao
import math
import pytest

audio_urls = [
    r"https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/ratemap/t29_lwwj2n_m17_lgwe7s.wav",
    ]


@pytest.fixture(params=audio_urls)
def audio_data(request):
    return ao.io.wave_read(request.param)


def test_gammatonegram(audio_data):
    data, sample_rate = audio_data
    frame_length = 10  # [ms]
    frame_samples = math.ceil(frame_length / 1000 * sample_rate)
    plot, ax = ao.plot.gammatonegram(
        data,
        sample_rate,
        frame_samples,
        num_features=64,
        low_Hz=50,
        high_Hz=8000
        )
