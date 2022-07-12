import ao

from gdrive import GDrive

import io
import os
import torch
import numpy as np
import pandas as pd
import webdataset as wds
import pytorch_lightning as pl

from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Callable, Tuple, List

CACHE_FOLDER = Path(__file__).parent.parent / 'datasets'
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


def _download_dataset_from_gdrive(
        folder_id: str, gdrive: Optional[GDrive]
    ) -> Path:
    if gdrive is None:
        gdrive = GDrive()
    folder = gdrive.drive.CreateFile({'id': folder_id})
    folder.FetchMetadata(fields='title')
    dataset_path = CACHE_FOLDER / folder['title']
    dataset_path.mkdir(exist_ok=True)
    to_download = []
    for f in gdrive.list_folder(folder_id):
        if ((
            f['title'] == 'dataset.yaml' or f['title'].endswith('.csv')
            or f['title'].endswith('.tar')
            ) and not (dataset_path / f['title']).exists()):
            to_download.append(f)
    if to_download:
        for f in tqdm(to_download, desc='Files', unit='file'):
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
        f"DATASETS_FOLDER enviroment variable is set to `{datasets_folder}`"
        "but it doesn't point to a valid local folder nor is a google drive "
        "folder url."
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


class WheelTestBedDataset(pl.LightningDataModule):

    def __init__(
        self,
        dataset: str,
        split_data: Callable[[pd.DataFrame, dict], Tuple[List[int], List[int],
                                                         List[int]]],
        datasets_folder: Optional[str] = os.getenv('DATASETS_FOLDER', None),
        get_label: Callable[[dict], torch.tensor] = lambda sample: sample,
        batch_size: int = 6,
        shuffle: int = 1E6,
        rng: Optional['random.Random'] = None,
        ):
        super().__init__()
        if datasets_folder is None:
            load_dotenv()
            datasets_folder = os.getenv('DATASETS_FOLDER', None)
        self.get_label = get_label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng
        # Download data if not already downloaded
        self.dataset_path = _get_dataset_path(dataset, datasets_folder)
        # Load configuration
        self.config = ao.io.yaml_load(self.dataset_path / 'dataset.yaml')
        self.config['name'] = self.dataset_path.name
        # Prepare data
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
        self.train_indices, self.val_indices, self.test_indices = split_data(
            self.data, self.config
            )

    @property
    def input_dim(self):
        return (
            len(self.config['extractors']),
            self.config['segment_frames'],
            self.config['frame_features'],
            )

    @property
    def sample_duration(self):
        return (
            self.config['segment_frames'] * self.config['frame_duration']
            ) / 1000

    @property
    def train_data(self):
        return self.data.iloc[self.train_indices]

    @property
    def val_data(self):
        return self.data.iloc[self.val_indices]

    @property
    def test_data(self):
        return self.data.iloc[self.test_indices]

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
            wds.batched(self.batch_size, partial=False),
            wds.shuffle(self.shuffle, rng=self.rng),
            )

    def train_dataloader(self):
        return self.get_dataloader(self.train_indices).with_length(
            int(len(self.train_indices) / self.batch_size)
            )

    def val_dataloader(self):
        return self.get_dataloader(self.val_indices).with_length(
            int(len(self.val_indices) / self.batch_size)
            )

    def test_dataloader(self):
        return self.get_dataloader(self.test_indices).with_length(
            int(len(self.test_indices) / self.batch_size)
            )


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