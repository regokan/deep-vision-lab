import os
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import Dataset, random_split
from torchvision import datasets


class DatasetWrapper:
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        root: str = "./data",
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        train_ratio: float = 0.8,
        download: bool = True,
    ):
        """
        A wrapper class for loading datasets either from torchvision or from local directories,
        with train, test, and validation splits and independent transforms.

        Args:
            dataset_name (Optional[str], optional): Name of the dataset to load from torchvision (e.g., 'MNIST').
                                                   If None, expects dataset to be in `root` folder with 'train', 'test', and optionally 'val' subfolders.
            root (str): Root directory for the dataset if loading from local folders.
            train_transform (Optional[Callable], optional): Transformations for training data.
            test_transform (Optional[Callable], optional): Transformations for test data.
            val_transform (Optional[Callable], optional): Transformations for validation data.
            train_ratio (float, optional): Proportion of training data used for training (if no validation set exists).
            download (bool, optional): Whether to download the dataset if not present (only applicable to torchvision datasets).

        Attributes:
            train_set (Dataset): The training dataset.
            val_set (Dataset): The validation dataset.
            test_set (Dataset): The test dataset.
        """
        self.root = root
        self.dataset_name = dataset_name
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.val_transform = val_transform
        self.train_ratio = train_ratio
        self.download = download

        # Initialize datasets
        self.train_set, self.val_set, self.test_set = self._load_datasets()

    def _get_torchvision_dataset(
        self, train: bool, transform: Optional[Callable]
    ) -> Dataset:
        """Load dataset directly from torchvision."""
        dataset_class = getattr(datasets, self.dataset_name)
        return dataset_class(
            root=self.root, train=train, transform=transform, download=self.download
        )

    def _load_datasets(
        self,
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """
        Load datasets, either from torchvision or from local folders.

        Returns:
            Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]: Train, validation, and test datasets.
        """
        if self.dataset_name:
            # Load dataset from torchvision
            train_set = self._get_torchvision_dataset(
                train=True, transform=self.train_transform
            )
            test_set = self._get_torchvision_dataset(
                train=False, transform=self.test_transform
            )

            # Split training data into training and validation if no predefined validation set
            train_len = int(len(train_set) * self.train_ratio)
            val_len = len(train_set) - train_len
            train_set, val_set = random_split(train_set, [train_len, val_len])

            # Apply validation transform if provided
            if self.val_transform:
                val_set.dataset.transform = self.val_transform
        else:
            # Load dataset from local folders
            train_set = datasets.ImageFolder(
                root=os.path.join(self.root, "train"), transform=self.train_transform
            )
            test_set = datasets.ImageFolder(
                root=os.path.join(self.root, "test"), transform=self.test_transform
            )

            # Check if a validation folder exists; otherwise, split training data
            val_set = None
            if os.path.isdir(os.path.join(self.root, "val")):
                val_set = datasets.ImageFolder(
                    root=os.path.join(self.root, "val"), transform=self.val_transform
                )
            else:
                train_len = int(len(train_set) * self.train_ratio)
                val_len = len(train_set) - train_len
                train_set, val_set = random_split(train_set, [train_len, val_len])

                # Apply validation transform if provided
                if self.val_transform:
                    val_set.dataset.transform = self.val_transform

        return train_set, val_set, test_set

    def get_datasets(
        self,
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        """Return the datasets for train, validation, and test."""
        return self.train_set, self.val_set, self.test_set
