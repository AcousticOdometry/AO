# TODO document
import ao

from gdrive import GDrive

import os
import webdataset as wds

from typing import Dict
from pathlib import Path
from dotenv import load_dotenv

CACHE_FOLDER = Path(__file__).parent.parent / 'datasets'
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


def get_config_from_filesystem(folder: Path) -> dict:
    config = {}
    for config_file in folder.glob('*.yaml'):
        file_params = ao.dataset.parse_filename(config_file.stem)
        key = next(iter(file_params))
        config.setdefault(key, {}).update({
            file_params[key]: ao.io.yaml_load(config_file)
            })
    return config


def get_shards_from_filesystem(folder: Path) -> Dict[str, dict]:
    return {
        f"file:{path}": ao.dataset.parse_filename(path.stem)
        for path in sorted(folder.glob('*.tar'))
        }


def get_config_from_gdrive(folder_id: str, gdrive: GDrive) -> dict:
    config = {}
    for f in gdrive.list_folder(folder_id):
        if f['title'].endswith('.yaml'):
            file_params = ao.dataset.parse_filename(
                f['title'].replace('.yaml', '')
                )
            key = next(iter(file_params))
            config.setdefault(key, {}).update({
                file_params[key]: gdrive.yaml_load(f)
                })
    return config


def get_shards_from_gdrive(folder_id: str, gdrive: GDrive) -> Dict[str, dict]:
    folder = gdrive.drive.CreateFile({'id': folder_id})
    folder.FetchMetadata(fields='title')
    dataset_folder = CACHE_FOLDER / folder['title']
    dataset_folder.mkdir(exist_ok=True)
    for f in gdrive.list_folder(folder_id):
        if f['title'].endswith('.tar'):
            # Check if cached in LOCAL_FOLDER
            if (dataset_folder / f['title']).exists():
                continue
            print(f"Downloading {folder['title']} shard {f['title']}...")
            f.GetContentFile(str(dataset_folder / f['title']))
    return get_shards_from_filesystem(dataset_folder)


def get_dataset_shards_and_config(dataset: str) -> Dict[str, Path]:
    # Full path provided
    dataset_path = Path(dataset)
    if dataset_path.is_absolute():
        return (
            get_shards_from_filesystem(dataset_path),
            get_config_from_filesystem(dataset_path)
            )
    # Full url provided
    gdrive_folder_id = GDrive.get_folder_id(dataset)
    if gdrive_folder_id:
        gdrive = GDrive()
        return (
            get_shards_from_gdrive(gdrive_folder_id, gdrive),
            get_config_from_gdrive(gdrive_folder_id, gdrive),
            )
    # Otherwise make use of DATASETS_FOLDER environment variable
    load_dotenv()
    datasets_folder = os.getenv('DATASETS_FOLDER')
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
        dataset_path = datasets_folder_path / dataset
        return (
            get_shards_from_filesystem(dataset_path),
            get_config_from_filesystem(dataset_path)
            )
    # Datasets folder is a gdrive url
    gdrive_folder_id = GDrive.get_folder_id(datasets_folder)
    if gdrive_folder_id:
        gdrive = GDrive()
        for folder in gdrive.list_folder(gdrive_folder_id):
            if folder['title'] == dataset:
                return (
                    get_shards_from_gdrive(folder['id'], gdrive),
                    get_config_from_gdrive(folder['id'], gdrive),
                    )
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

if __name__ == "__main__":
    import io
    import json
    import numpy as np

    from argparse import ArgumentParser

    parser = ArgumentParser("Test accessibility to a WebDataset")
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()
    print(f"Getting `{args.dataset}` shard list and configuration...")
    shards, config = get_dataset_shards_and_config(args.dataset)
    print('config = ', ao.io.yaml_dump(config))
    print('shards = ', json.dumps(shards, indent=2))
    print("First sample of WebDataset...")
    dataset = wds.DataPipeline(
        wds.SimpleShardList(list(shards.keys())),
        wds.tarfile_to_samples(),
        wds.decode(
            wds.handle_extension('.npy', lambda x: np.load(io.BytesIO(x)))
            ),
        wds.to_tuple('npy', 'json'),
        )
    for features, result in dataset:
        print(f"{features.shape = }")
        print(f"{result = }")
        
