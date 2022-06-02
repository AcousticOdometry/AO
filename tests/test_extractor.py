import ao

import math
import warnings
import numpy as np

# TODO test overide Extractor

def test_extractor():
    audio_url = (
        r"https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources"
        r"/ratemap/t29_lwwj2n_m17_lgwe7s.wav"
        )
    data, fs = ao.io.wave_read(audio_url)
    frame_length = 10  # [ms]
    num_samples = int(np.ceil(frame_length / 1000 * fs))  # samples per frame
    num_frames = math.ceil(data.size / num_samples)
    padded_data = np.append(
        data, np.zeros(num_frames * num_samples - data.size)
        )
    extractor = ao.extractor.GammatoneFilterbank(num_samples, 64, fs)
    for index in range(num_frames):
        batch = padded_data[index * num_samples:(index + 1) * num_samples]
        output = extractor(batch)
        # TODO check output size and that there are no NaN or Inf

def test_gammatone_filterbank():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        extractor = ao.extractor.GammatoneFilterbank()
        for filter in extractor.filters:
            assert isinstance(filter.cf, float)
            assert isinstance(filter.gain, float)
            assert isinstance(filter.a, list)
        
