import ao
import re
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List


def _get_regression_Vx(samples: np.ndarray, model: ao.AO):
    return model.predict(samples).item()


def _get_classification_Vx(
        samples: np.ndarray, model: ao.AO, centers: List[float]
    ):
    return centers[int(model.predict(samples).argmax(1).sum().item())]


def _get_ordinal_classification_Vx(
        samples: np.ndarray, model: ao.AO, centers: List[float]
    ):
    prediction = model.predict(samples)
    label = (prediction > 0.5).cumprod(axis=1).sum(axis=1) - 1
    return centers[int(max(label.item(), 0))]


def get_extractors(dataset_config: dict, sample_rate: int):
    # Build extractors
    extractors = []
    for extractor in dataset_config['extractors']:
        # Get the extractor class
        try:
            extractor_class = getattr(ao.extractor, extractor['class'])
        except AttributeError as e:
            raise RuntimeError(
                f"Extractor `{extractor['class']}` not found: {e}"
                )
        # Get the extractor keyword arguments
        _kwargs = {}
        for key, arg in extractor['kwargs'].items():
            # Check if the argument is a function source
            if isinstance(arg, str):
                match = re.search(r"def (?P<function_name>.*)\(", arg)
                if match:
                    # If it is function source, execute it and get the function
                    # handle
                    # ! This code is unsafe, it could be tricked by a
                    # ! maliciously built dataset_config file
                    exec(arg)
                    _kwargs[key] = eval(match.group('function_name'))
                    continue
            _kwargs[key] = arg
        extractors.append(
            extractor_class(
                num_samples=int(
                    dataset_config['frame_duration'] * sample_rate / 1000
                    ),
                num_features=dataset_config['frame_features'],
                sample_rate=sample_rate,
                **_kwargs
                )
            )
    return extractors


def generate_file_acoustic_odometry(
    wav_file: Path,
    model: 'pl.LightningModule',
    dataset_config: dict,
    get_Vx: callable,
    ):
    wav_data, sample_rate = ao.io.audio_read(wav_file)
    extractors = get_extractors(dataset_config, sample_rate)
    frames = ao.dataset.audio._frames(wav_data, extractors[0].num_samples)
    Vx = np.empty(len(frames))
    features = torch.zeros(
        [
            1,
            len(extractors), extractors[0].num_features,
            dataset_config['segment_frames']
            ],
        device=model.device,
        )
    for i, frame in enumerate(frames):
        # Update features
        for k, extractor in enumerate(extractors):
            features[0, k, :, 0] = torch.as_tensor(extractor(frame))
        features = features.roll(-1, 3)
        # Predict and store
        prediction = model(features)
        Vx[i] = get_Vx(prediction)
    # Compute X translations and cumulative X position
    config = ao.io.yaml_load(wav_file.with_suffix('.yaml'))
    start = config['start_timestamp']
    step = extractors[0].num_samples / sample_rate
    timestamps = np.linspace(start, start + len(frames) * step, len(frames))
    Vx = pd.Series(Vx, index=timestamps)
    odom = pd.concat([Vx, Vx.index.to_series().diff() * Vx], axis=1)
    odom.columns = ['Vx', 'tx']
    odom.iloc[0, :] = 0
    odom['X'] = odom['tx'].cumsum()
    return odom


def get_recording_evaluation(
    recording: Path,
    model: 'pl.LightningModule',
    extractors: List[ao.extractor.Extractor],
    ):
    gt = pd.read_csv(recording / 'ground_truth.csv', index_col='timestamps')
    # TODO Evaluate with ground truth?
