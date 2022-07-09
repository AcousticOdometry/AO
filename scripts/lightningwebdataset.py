from dataset import get_dataset_shards_and_config

import io
import torch
import random
import logging
import numpy as np
import webdataset as wds
import pytorch_lightning as pl

from pathlib import Path
from datetime import datetime
from functools import partial
from typing import Dict, List, Tuple

logging_folder = Path().parent / 'logs'
logging_folder.mkdir(exist_ok=True)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            logging_folder / datetime.now().strftime('%Y%m%d_%H%M%S.log')
            ),
        stream_handler,
        ],
    level=logging.DEBUG
    )


def _select_shards_by_transform_and_device(
    shards: Dict[str, dict],
    use_transforms: List[str] = ['None'],
    test_devices: List[str] = ['rode-videomic-ntg-top', 'rode-smartlav-top'],
    ) -> Tuple[List[str], List[str]]:
    train = []
    test = []
    for url, params in shards.items():
        if params['transform'] not in use_transforms:
            continue
        elif params['device'] in test_devices:
            test.append(url)
        else:
            train.append(url)
    return train, test


SHARD_SELECTION_STRATEGIES = {
    'base':
        partial(
            _select_shards_by_transform_and_device,
            use_transforms=['None'],
            test_devices=['rode-videomic-ntg-top', 'rode-smartlav-top'],
            ),
    'random-noise':
        partial(
            _select_shards_by_transform_and_device,
            use_transforms=['None', 'add-random-snr-noise'],
            test_devices=['rode-videomic-ntg-top', 'rode-smartlav-top'],
            ),
    'all-devices':
        partial(
            _select_shards_by_transform_and_device,
            use_transforms=['None'],
            test_devices=[],
            ),
    }


class LightningWebDataset(pl.LightningDataModule):

    def __init__(
            self,
            dataset: str,
            shard_selection_strategy: str = 'base',
            validation_split: float = 0.2,
            batch_size: int = 6,
            seed: int = random.randint(0, 2**32),
            shuffle: int = 1E6,
        ):
        super().__init__()
        # Check hyperparameters
        if validation_split >= 1 or validation_split < 0:
            raise ValueError(
                "`validation_split` must be in range [0,1[ but is "
                f"{validation_split}"
                )
        self.validation_split = validation_split
        if shard_selection_strategy not in SHARD_SELECTION_STRATEGIES:
            raise ValueError(
                "Invalid shard selection strategy. Available options: "
                f"{list(SHARD_SELECTION_STRATEGIES.keys())}"
                )
        self.select_shards = SHARD_SELECTION_STRATEGIES[
            shard_selection_strategy]
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        # Applied to features
        # TODO regression or classification
        # TODO Vx or slip + Vw ?
        # self.get_label = lambda result: torch.tensor(int(result['Vx'] * 100))
        # TODO self.dims
        # self.num_classes
        # Random number generators for splitting train and test samples
        self._prepare_data(dataset)

    def _prepare_data(self, dataset) -> None:
        # Get and split shards
        shards, self.config = get_dataset_shards_and_config(dataset)
        self.train_shards, self.test_shards = self.select_shards(shards)
        # Compute shard sample lengths
        self.train_length = sum([
            self.config['shards'][shards[shard]['name']]['count']
            for shard in self.train_shards
            ])
        self.test_length = sum([
            self.config['shards'][shards[shard]['name']]['count']
            for shard in self.test_shards
            ])
        logging.info(
            f"Train: {self.train_length} samples, "
            f"{self.train_length / self.batch_size} batches"
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

    def get_features(self, array):
        return torch.from_numpy(array)

    def get_label(self, result):
        return round(result['Vx'] * 100)

    def _get_dataset(self, shards):
        return wds.DataPipeline(
            wds.SimpleShardList(shards),
            wds.tarfile_to_samples(),
            wds.shuffle(self.shuffle),
            wds.decode(
                wds.handle_extension('.npy', lambda x: np.load(io.BytesIO(x)))
                ),
            wds.to_tuple('npy', 'json'),
            wds.map_tuple(self.get_features, self.get_label),
            # wds.batched(self.batch_size, partial=False),
            )

    def train_dataloader(self):
        logging.info('train_dataloader()')
        dataset = self._get_dataset(self.train_shards).with_length(
            self.train_length / self.batch_size
            )
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        logging.info('val_dataloader()')
        return None

    def test_dataloader(self):
        if self.test_shards:
            return self._get_dataset(shards=self.test_shards).with_length(
                self.test_length / self.batch_size
                )
        return None


if __name__ == "__main__":
    import io
    import numpy as np

    from argparse import ArgumentParser

    parser = ArgumentParser("Test WebDataset creation and splits")
    parser.add_argument('dataset', type=str)
    # parser.add_argument('--total', type=int, default=10)
    parser.add_argument('-s', '--validation-split', type=float, default=0.2)
    args = parser.parse_args()
    dataset = LightningWebDataset(
        args.dataset, validation_split=args.validation_split
        )
    print(
        f"Checking dataset {args.dataset} with test split "
        f"{args.validation_split}"
        )
    # Save samples to list
    val_samples = list(dataset.val_dataloader())
    # Train samples should never be in the test set
    train = 0
    for features, _ in dataset.train_dataloader():
        for val_features, _ in val_samples:
            assert not torch.equal(
                features, val_features
                ), "Train sample in validation set"
        train += 1
    validation = len(val_samples)
    total = validation + train
    print(f"Train: {train}/{total} = {train/total}")
    print(f"Validation: {validation}/{total} = {validation/total}")