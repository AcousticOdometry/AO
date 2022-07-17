import ao

from gdrive import GDrive

import io
import os
import re
import torch
import tempfile
import numpy as np
import pandas as pd
import webdataset as wds
import pytorch_lightning as pl

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Callable, Tuple, List

LOCAL_DATASETS_FOLDER = Path(__file__).parent.parent / 'datasets'
LOCAL_DATASETS_FOLDER.mkdir(parents=True, exist_ok=True)
LOCAL_EVALUATION_FOLDER = Path(__file__).parent.parent / 'evaluation'
LOCAL_EVALUATION_FOLDER.mkdir(parents=True, exist_ok=True)


def _get_evaluation_folder(evaluation_folder: str) -> Path:
    # Full path provided
    evaluation_path = Path(evaluation_folder)
    if evaluation_path.is_absolute():
        return evaluation_path
    # Full url provided
    folder_id = GDrive.get_folder_id(evaluation_folder)
    if not folder_id:
        raise ValueError(
            f"{evaluation_folder = } is not a valid path or google drive url"
            )
    gdrive = GDrive()
    evaluation_folder = LOCAL_EVALUATION_FOLDER
    to_download = []
    for r in gdrive.list_folder(folder_id):
        recording = evaluation_folder / r['title']
        recording.mkdir(exist_ok=True)
        for f in gdrive.list_folder(r['id']):
            if not (
                f['title'].endswith('.wav') or f['title'].endswith('.yaml')
                or f['title'].endswith('.csv')
                ):
                continue
            elif (recording / f['title']).exists():
                continue
            to_download.append((recording / f['title'], f))
    if to_download:
        for path, gfile in tqdm(to_download, desc='Evaluation', unit='file'):
            gdrive.download_file(gfile, path)
    return evaluation_folder


def _download_dataset_from_gdrive(
        folder_id: str, gdrive: Optional[GDrive]
    ) -> Path:
    if gdrive is None:
        gdrive = GDrive()
    folder = gdrive.drive.CreateFile({'id': folder_id})
    folder.FetchMetadata(fields='title')
    dataset_path = LOCAL_DATASETS_FOLDER / folder['title']
    dataset_path.mkdir(exist_ok=True)
    to_download = []
    for f in gdrive.list_folder(folder_id):
        if ((
            f['title'] == 'dataset.yaml' or f['title'].endswith('.csv')
            or f['title'].endswith('.tar')
            ) and not (dataset_path / f['title']).exists()):
            to_download.append(f)
    if to_download:
        for f in tqdm(to_download, desc='Dataset', unit='file'):
            gdrive.download_file(f, dataset_path / f['title'])
    return dataset_path


def _get_dataset_path(dataset: str, datasets_folder: str) -> Path:
    # Full path provided
    dataset_path = Path(dataset)
    if dataset_path.is_absolute():
        return dataset_path
    # Full url provided
    gdrive_folder_id = GDrive.get_folder_id(dataset)
    if gdrive_folder_id:
        return _download_dataset_from_gdrive(gdrive_folder_id)
    # Otherwise make use of datasets_folder
    if datasets_folder is None:
        raise ValueError(
            "Dataset full path can't be resolved. Provide a the full local "
            f"path of a dataset (instead of `{dataset}`) or set the "
            "DATASETS_FOLDER environment variable to a local folder or a "
            "google drive folder url."
            )
    # Datasets folder is an absolute path
    datasets_folder_path = Path(datasets_folder)
    if datasets_folder_path.is_absolute():
        return datasets_folder_path / dataset
    # Datasets folder is a gdrive url
    gdrive_folder_id = GDrive.get_folder_id(datasets_folder)
    if gdrive_folder_id:
        gdrive = GDrive()
        for folder in gdrive.list_folder(gdrive_folder_id):
            if folder['title'] == dataset:
                return _download_dataset_from_gdrive(folder['id'], gdrive)
        raise ValueError(
            f"Dataset `{dataset}` not found in google drive folder "
            f"{datasets_folder}. Expected a folder named as the dataset."
            )
    # Can't resolve dataset path
    raise ValueError(
        f"Provided `{datasets_folder}` doesn't point to a valid local folder "
        "nor is a google drive folder url."
        )


def _subset_samples(data, indices):
    yielded = 0
    for sample_n, sample in enumerate(data):
        # Assumes indices are sorted
        try:
            if indices[yielded] == sample_n:
                yielded += 1
                yield sample
        except IndexError:  # There is data but indices are finished
            break


subset_samples = wds.filters.pipelinefilter(_subset_samples)


def _decode_npy(_bytes):
    return np.load(io.BytesIO(_bytes))


class RecordingsDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        recordings: List[Path],
        config: dict,
        file_filter: Callable[[dict], bool] = lambda device_config: True,
        ):
        super().__init__()
        self.recordings = recordings
        self.config = config
        self.file_filter = file_filter

    def __iter__(self):
        for recording in self.recordings:
            gt = pd.read_csv(
                recording / 'ground_truth.csv', index_col='timestamp'
                )
            for wav_file in sorted(recording.glob('*.wav')):
                file_config = ao.io.yaml_load(wav_file.with_suffix('.yaml'))
                if not self.file_filter(file_config):
                    continue
                yield {
                    'recording': recording.name,
                    'file': wav_file.name,
                    'features': self.features(wav_file),
                    'start_timestamp': file_config['start_timestamp'],
                    'frame_duration': self.config['frame_duration'] / 1000.0,
                    'ground_truth': gt,
                    }

    def features(self, wav_file: Path):
        try:
            return np.load(
                wav_file.parent / self.config['name'] /
                wav_file.with_suffix('.npy').name
                )
        except FileNotFoundError:
            wav_data, sample_rate = ao.io.audio_read(wav_file)
            extractors = self.extractors(sample_rate)
            features = ao.dataset.audio.features(wav_data, extractors)
            folder = wav_file.parent / self.config['name']
            folder.mkdir(exist_ok=True)
            np.save(folder / wav_file.with_suffix('.npy').name, features)
            return features

    _extractors = {}

    def extractors(self, sample_rate: int):
        try:
            return self._extractors[sample_rate]
        except KeyError:
            self._extractors[sample_rate] = self._build_extractors(
                self.config, sample_rate
                )
            return self._extractors[sample_rate]

    @staticmethod
    def _build_extractors(dataset_config: dict, sample_rate: int):
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
                        # If it is function source, execute it and get the
                        # function handle
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


class WheelTestBedDataset(pl.LightningDataModule):

    def __init__(
        self,
        dataset: str,
        split_data: Callable[[pd.DataFrame, dict], Tuple[List[int], List[int],
                                                         List[int]]],
        datasets_folder: Optional[str] = os.getenv('DATASETS_FOLDER', None),
        evaluation_folder: Optional[str] = os.getenv(
            'EVALUATION_FOLDER', None
            ),
        get_label: Callable[[dict], torch.tensor] = lambda sample: sample,
        batch_size: int = 6,
        shuffle: int = 1E4,
        rng: Optional['random.Random'] = None,
        ):
        super().__init__()
        if datasets_folder is None:
            load_dotenv()
            datasets_folder = os.environ['DATASETS_FOLDER']
        if evaluation_folder is None:
            load_dotenv()
            evaluation_folder = os.environ['EVALUATION_FOLDER']
        self.get_label = get_label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng
        # Download data if not already downloaded
        self.dataset_path = _get_dataset_path(dataset, datasets_folder)
        # Load configuration
        self.config = ao.io.yaml_load(self.dataset_path / 'dataset.yaml')
        self.config['name'] = self.dataset_path.name
        # Prepare training data
        self.shards = [
            f"file:{self.dataset_path / s_name}"
            for s_name in self.config['shards'].keys()
            ]
        self.data = pd.concat(
            [
                pd.read_csv(f).assign(**ao.dataset.parse_filename(f.stem))
                for f in self.dataset_path.glob('*.csv')
                ],
            ignore_index=True,
            )
        self.train_indices, _, _ = split_data(self.data, self.config)
        # Prepare evaluation data
        self.evaluation_folder = _get_evaluation_folder(evaluation_folder)
        self.test_recordings = sorted([
            r for r in self.evaluation_folder.iterdir() if r.is_dir()
            ])
        self.val_recordings = self.test_recordings[0:1]
        # ? Input parameter ?
        self.val_file_filter = lambda device: 'VideoMic' in device['name']

    @property
    def input_dim(self):
        return (
            len(self.config['extractors']),
            self.config['frame_features'],
            self.config['segment_frames'],
            )

    @property
    def sample_duration(self):
        return (
            self.config['segment_frames'] * self.config['frame_duration']
            ) / 1000

    @property
    def train_data(self):
        return self.data.iloc[self.train_indices]

    def get_features(self, features: np.ndarray):
        return torch.from_numpy(features)

    def get_dataloader(self, indices):
        if not indices:
            return None
        return wds.DataPipeline(
            wds.SimpleShardList(self.shards),
            wds.tarfile_to_samples(),
            subset_samples(indices),  # Filters the samples
            wds.decode(wds.handle_extension('.npy', _decode_npy)),
            wds.to_tuple('npy', 'json'),
            wds.map_tuple(self.get_features, self.get_label),
            wds.shuffle(self.shuffle, rng=self.rng),
            wds.batched(self.batch_size, partial=False),
            ).with_length(int(len(indices) / self.batch_size))

    def train_dataloader(self):
        return self.get_dataloader(self.train_indices)

    def val_dataloader(self):
        return RecordingsDataset(
            self.val_recordings, self.config, self.val_file_filter
            )

    def test_dataloader(self):
        return RecordingsDataset(self.test_recordings, self.config)


if __name__ == "__main__":
    import random
    from argparse import ArgumentParser

    parser = ArgumentParser("Test accessibility to a WheelTestBedDataset")
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    print(f"Using `{args.dataset}`...")

    def split_data(data):
        train_indices, val_indices, test_indices = [], [], []
        for index, sample in data.iterrows():
            if sample['transform'] != 'None':
                continue
            if sample['device'] in [
                'rode-videomic-ntg-top', 'rode-smartlav-top'
                ]:
                test_indices.append(index)
            elif random.uniform(0, 1) < 0.8:
                train_indices.append(index)
            else:
                val_indices.append(index)
        return (train_indices, val_indices, test_indices)

    dataset = WheelTestBedDataset(args.dataset, split_data, shuffle=0)
    print('config = ', ao.io.yaml_dump(dataset.config))

    train_dataset = dataset.train_dataloader()
    train_samples = 0
    print('Starting test')
    for batch_features, batch_samples in train_dataset:
        for sample in batch_samples:
            if not np.isclose(
                dataset.train_data.iloc[train_samples]['Vx'], sample['Vx']
                ):
                print(train_samples)
                print(dataset.train_data.iloc[train_samples])
                print(sample['Vx'])
                raise RuntimeError
            train_samples += 1
    if train_samples != len(dataset.train_indices):
        raise AssertionError(
            f"{train_samples = } != {len(dataset.train_indices) = }"
            )