import ao

def test_ao(audio_data, model_folder):
    model = ao.AO.new(model_folder)
    data, _ = audio_data
    for i, frame in enumerate(
        ao.dataset.audio._frames(data, model.num_samples)
        ):
        features_1 = model.features
        result = model(frame)
        print(model.prediction)
        if result > 0:
            print(result)
        assert features_1[0, :, :, 1:].equal(model.features[0, :, :, 0:-1])
        if i > 50:
            break