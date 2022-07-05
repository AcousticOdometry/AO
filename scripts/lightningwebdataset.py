from dataset import get_dataset_shards_and_config

import io
import torch
import random
import numpy as np
import webdataset as wds
import pytorch_lightning as pl

from functools import partial
from typing import Dict, List, Tuple


def _select_shards_by_transform_and_device(
    shards: Dict[str, dict],
    use_transforms: List[str] = ['None'],
    validation_devices: List[str] = [
        'rode-videomic-ntg-top', 'rode-smartlav-top'
        ],
    ) -> Tuple[List[str], List[str]]:
    train = []
    validation = []
    for url, params in shards.items():
        if params['transform'] not in use_transforms:
            continue
        elif params['device'] in validation_devices:
            validation.append(url)
        else:
            train.append(url)
    return train, validation


SHARD_SELECTION_STRATEGIES = {
    'base':
        partial(
            _select_shards_by_transform_and_device,
            use_transforms=['None'],
            validation_devices=['rode-videomic-ntg-top', 'rode-smartlav-top'],
            ),
    }


def _split_train(data, test_split, random_seed):
    assert 0.0 <= test_split <= 1.0
    rng = random.Random(random_seed)
    for sample in data:
        if rng.uniform(0.0, 1.0) >= test_split:
            yield sample


def _split_test(data, test_split, random_seed):
    assert 0.0 <= test_split <= 1.0
    rng = random.Random(random_seed)
    for sample in data:
        if rng.uniform(0.0, 1.0) < test_split:
            yield sample


split_train = wds.filters.pipelinefilter(_split_train)
split_test = wds.filters.pipelinefilter(_split_test)


class LightningWebDataset(pl.LightningDataModule):

    def __init__(
        self,
        dataset: str,
        shard_selection_strategy: str = 'base',
        test_split: float = 0.2,
        batch_size: int = 6,
        # include_transforms: List[str] = [],
        seed: int = random.randint(0, 2**32),
        shuffle: int = 1E6,
        ):
        super().__init__()
        shards, self.config = get_dataset_shards_and_config(dataset)
        if shard_selection_strategy not in SHARD_SELECTION_STRATEGIES:
            raise ValueError(
                "Invalid shard selection strategy. Available options: "
                f"{list(SHARD_SELECTION_STRATEGIES.keys())}"
                )
        select_shards = SHARD_SELECTION_STRATEGIES[shard_selection_strategy]
        self.train_shards, self.validation_shards = select_shards(shards)
        if test_split >= 1 or test_split < 0:
            raise ValueError(
                f"`test_split` must be in range [0,1[ but is {test_split}"
                )
        self.test_split = test_split
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        # Applied to features
        self.get_batch_features = torch.from_numpy
        # TODO regression or classification
        # TODO Vx or slip + Vw ?
        # self.get_label = lambda result: torch.tensor(int(result['Vx'] * 100))
        # TODO self.dims
        # self.num_classes
        # Random number generators for splitting train and test samples

    def get_batch_labels(self, batch):
        return torch.tensor([int(result['Vx'] * 100) for result in batch])

    def get_dataloader(self, shards, *filter_shards):
        return wds.DataPipeline(
            wds.SimpleShardList(shards),
            wds.tarfile_to_samples(),
            wds.decode(
                wds.handle_extension('.npy', lambda x: np.load(io.BytesIO(x)))
                ),
            wds.to_tuple('npy', 'json'),
            wds.batched(self.batch_size, partial=False),
            wds.map_tuple(self.get_batch_features, self.get_batch_labels),
            *filter_shards,
            wds.shuffle(self.shuffle),
            )

    def train_dataloader(self):
        return self.get_dataloader(
            self.train_shards, split_train(self.test_split, self.seed)
            )

    def test_dataloader(self):
        return self.get_dataloader(
            self.train_shards, split_test(self.test_split, self.seed)
            )

    def val_dataloader(self):
        return self.get_dataloader(shards=self.validation_shards)


if __name__ == "__main__":
    import io
    import numpy as np

    from argparse import ArgumentParser

    parser = ArgumentParser("Test WebDataset creation and splits")
    parser.add_argument('dataset', type=str)
    # parser.add_argument('--total', type=int, default=10)
    parser.add_argument('--test-split', type=float, default=0.2)
    args = parser.parse_args()
    dataset = LightningWebDataset(args.dataset, test_split=args.test_split)
    # all_samples = iter(dataset.get_dataloader(dataset.train_shards))
    train_samples = iter(dataset.train_dataloader())
    test_samples = iter(dataset.test_dataloader())

    print(f"Checking dataset {args.dataset} with test split {args.test_split}")
    # Save samples to list
    test_samples = list(test_samples)
    # Train samples should never be in the test set
    train = 0
    for features, _ in train_samples:
        for test_features, _ in test_samples:
            assert not torch.equal(
                features, test_features
                ), "Train sample in test set"
        train += 1
    test = len(test_samples)
    total = test + train
    print(f"Train: {train}/{total} = {train/total}")
    print(f"Test: {test}/{total} = {test/total}")


    # Test valid only when not shuffling
    # train = 0
    # test = 0
    # print(
    #     f"Extracting {args.total} samples with a test split of "
    #     f"{args.test_split}"
    #     )
    # next_train_sample, _ = next(train_samples)
    # next_test_sample, _ = next(test_samples)
    # for total in range(args.total):
    #     try:
    #         sample, _ = next(all_samples)
    #         if torch.equal(sample, next_train_sample):
    #             train += 1
    #             next_train_sample, _ = next(train_samples)
    #         elif torch.equal(sample, next_test_sample):
    #             test += 1
    #             next_test_sample, _ = next(test_samples)
    #         else:
    #             print(sample[0, 0, 0, 0].item())
    #             raise RuntimeError("Sample is nor train nor test")
    #     except StopIteration:
    #         break
    # total += 1
    # print(f"Train: {train}/{total} = {train/total}")
    # print(f"Test: {test}/{total} = {test/total}")
