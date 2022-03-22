import ao

import math
import numpy as np

# TODO test overide Extractor

# TODO test help(Extractor)


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
    extractor = ao.GammatoneFilterbank(num_samples, 64, fs)
    for index in range(num_frames):
        batch = padded_data[index * num_samples:(index + 1) * num_samples]
        output = extractor(batch)
        print([math.log10(x) for x in output])
