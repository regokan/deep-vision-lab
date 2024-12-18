import math
import multiprocessing
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms as T

from data import DataLoaderWrapper, DatasetWrapper
from data.helper import calculate_mean_std
from utils.device import get_device

from .helpers import get_data_location

device = get_device()


def get_data_loaders(
    batch_size: int = 32,
    valid_size: float = 0.2,
    num_workers: int = -1,
    limit: int = -1,
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """
    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    base_path = Path(get_data_location())
    root_dir = os.path.join(base_path, "landmark_images")

    # Initial data transformation
    initial_transform = T.Compose(
        [
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )

    # Load datasets
    dataset_wrapper = DatasetWrapper(
        root=root_dir,
        train_transform=initial_transform,
        download=True,
    )
    train_dataset, _, _ = dataset_wrapper.get_datasets()

    # Compute mean and std of the dataset
    mean, std = calculate_mean_std(train_dataset, device=device, batch_size=batch_size)
    print(f"Dataset mean: {mean}, std: {std}")

    # create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    # HINT: resize the image to 256 first, then crop them to 224, then add the
    # appropriate transforms for that step
    data_transforms = {
        "train": T.Compose(
            [
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        ),
        "valid": T.Compose(
            [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)]
        ),
        "test": T.Compose(
            [T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean, std)]
        ),
    }

    # Apply transforms to datasets
    data = DatasetWrapper(
        root=root_dir,
        train_transform=data_transforms["train"],
        test_transform=data_transforms["test"],
        val_transform=data_transforms["valid"],
        train_ratio=1 - valid_size,
        download=True,
    )
    train_data, valid_data, test_data = data.get_datasets()

    # Get lengths and validate limit usage
    n_train = len(train_data)
    n_valid = len(valid_data)
    n_test = len(test_data)

    if limit > 0:
        n_train = min(n_train, limit)
        n_valid = min(n_valid, limit)
        n_test = min(n_test, limit)

    # Define samplers for train/validation datasets
    train_indices = torch.randperm(n_train).tolist()
    valid_indices = torch.randperm(n_valid).tolist()

    # Use samplers within dataset limits
    train_sampler = SubsetRandomSampler(train_indices[:n_train])
    valid_sampler = SubsetRandomSampler(valid_indices[:n_valid])

    # Initialize DataLoaders with debug info
    data_loaders = {
        "train": DataLoaderWrapper(
            train_data,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
        ).get_loader(),
        "valid": DataLoaderWrapper(
            valid_data,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
        ).get_loader(),
        "test": DataLoaderWrapper(
            test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
        ).get_loader(),
    }

    print(
        f"Training set size: {n_train}, Validation set size: {n_valid}, Test set size: {n_test}"
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    # obtain one batch of training images
    # First obtain an iterator from the train dataloader
    dataiter = iter(data_loaders["train"])
    # Then call the .next() method on the iterator you just
    # obtained
    images, labels = next(dataiter)
    train_dataset = data_loaders["train"].dataset
    # Undo the normalization (for visualization purposes)
    mean, std = calculate_mean_std(train_dataset, device=device)
    invTrans = T.Compose(
        [
            T.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            T.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)
    # Get class names from the train data loader
    class_names = train_dataset.dataset.classes

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(class_names[labels[idx].item()])


if __name__ == "__main__":
    data_loaders = get_data_loaders(batch_size=2, num_workers=0)
    visualize_one_batch(data_loaders, max_n=2)
