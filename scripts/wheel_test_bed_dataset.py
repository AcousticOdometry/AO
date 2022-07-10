import ao

from gdrive import GDrive

import os
import torch
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from functools import partial
from dotenv import load_dotenv
from typing import Optional, Callable, Union, Dict, List

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
            f['title'] == 'data.zip' and not (dataset_path / 'data').exists()
            )
            or ((f['title'] == 'dataset.yaml' or f['title'].endswith('.csv'))
                and not (dataset_path / f['title']).exists())):
            to_download.append(f)
    for shard in to_download:
        gdrive.download_file(shard, dataset_path / shard['title'])
    if (dataset_path / 'data.zip').exists():
        if (dataset_path / 'data').exists():
            raise RuntimeError(
                f"Found compressed and uncompressed data in {dataset_path}"
                )
        shutil.unpack_archive(dataset_path / 'data.zip', dataset_path / 'data')
        (dataset_path / 'data.zip').unlink()
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


class WheelTestBedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset: str,
        datasets_folder: Optional[str] = os.getenv('DATASETS_FOLDER', None),
        sample_filter: Callable[[dict], bool] = lambda sample: True,
        label_from_sample: Callable[[pd.Series], int] = lambda sample: sample,
        ):
        if datasets_folder is None:
            load_dotenv()
            datasets_folder = os.getenv('DATASETS_FOLDER', None)
        dataset_path = _get_dataset_path(dataset, datasets_folder)
        # Load configuration
        self.config = ao.io.yaml_load(dataset_path / 'dataset.yaml')
        # Load and filter data list
        self.data = pd.concat(
            [
                pd.read_csv(f).assign(**ao.dataset.parse_filename(f.stem))
                for f in dataset_path.glob('*.csv')
                ],
            ignore_index=True,
            )
        self.data = self.data[self.data.apply(sample_filter, axis=1)]
        if self.data.empty:
            raise RuntimeError('No data found that matched the filter')
        self.data.reset_index(inplace=True, drop=True)
        # Locate the features folder
        self.features_folder = dataset_path / 'data'
        self.label_from_sample = label_from_sample

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

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        features = np.load(
            self.features_folder / self.get_sample_filename(sample)
            )
        return torch.from_numpy(features), self.label_from_sample(sample)

    @staticmethod
    def get_sample_filename(sample: Union[dict, pd.Series]) -> str:
        return (
            f"transform_{sample['transform']};device_{sample['device']};"
            f"recording_{sample['recording']};segment_{sample['segment']}.npy"
            )


def _filter_by_transform_and_device(
    sample: pd.Series,
    use_transforms: List[str] = ['None'],
    use_devices: List[str] = ['rode-videomic-ntg-top', 'rode-smartlav-top'],
    ):
    return (
        sample['transform'] in use_transforms
        or sample['device'] in use_devices
        )


TRAIN_SPLITS = {
    'base': (
        partial(
            _filter_by_transform_and_device,
            use_transforms=['None'],
            use_devices=[
                'rode-videomic-ntg-back', 'rode-videomic-ntg-front',
                'rode-smartlav-wheel', 'laptop-built-in-microphone'
                ],
            ),
        partial(
            _filter_by_transform_and_device,
            use_transforms=['None'],
            use_devices=['rode-videomic-ntg-top', 'rode-smartlav-top'],
            ),
        )
    }


# TODO pytorch DataModule
def get_dataloaders(
    dataset: str,
    label_from_sample: Callable[[pd.Series], int] = lambda sample: sample,
    train_split: float = 0.8,
    batch_size: int = 15,
    datasets_folder: Optional[str] = os.getenv('DATASETS_FOLDER', None),
    ) -> Dict[str, torch.utils.data.DataLoader]:
    train_filter, test_filter = TRAIN_SPLITS['base']
    datasets = {
        'test':
            WheelTestBedDataset(
                dataset,
                sample_filter=test_filter,
                datasets_folder=datasets_folder,
                label_from_sample=label_from_sample
                )
        }
    train_dataset = WheelTestBedDataset(
        dataset,
        sample_filter=train_filter,
        datasets_folder=datasets_folder,
        label_from_sample=label_from_sample
        )
    train_size = int(train_split * len(train_dataset))
    test_size = len(train_dataset) - train_size
    datasets['train'], datasets['val'] = torch.utils.data.random_split(
        train_dataset, [train_size, test_size]
        )
    loaders = {}
    for name, dataset in datasets.items():
        loaders[name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if 'train' in name else False,
            )
        print(
            f'{name} set: {len(dataset)} samples, {len(loaders[name])} batches'
            )
    return loaders


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Test accessibility to a WebDataset")
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    print(f"Using `{args.dataset}`...")
    dataset = WheelTestBedDataset(args.dataset)
    print('config = ', ao.io.yaml_dump(dataset.config))
    print("First sample: ")
    features, result = next(iter(dataset))
    print(f"{features.shape = }")
    print(f"{result = }")