import ao

import numpy as np


def test_ao(audio_data, model_path):
    extractors = [
        ao.extractor.GammatoneFilterbank(
            num_samples=int(10 * 44100 / 1000),
            num_features=256,
            sample_rate=44100,
            )
        ]
    model = ao.AO(
        model_path=model_path,
        extractors=extractors,
        num_frames=120,
        )
    data, _ = audio_data
    for i, frame in enumerate(ao.dataset.audio._frames(
        data.mean(axis=1)[:, np.newaxis], model.num_samples
        )):
        features_1 = model.features
        model(frame)
        assert features_1[0, :, :, 1:].equal(model.features[0, :, :, 0:-1])
        if i > 50:
            break