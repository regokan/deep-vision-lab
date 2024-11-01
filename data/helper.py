from typing import Tuple

import torch
from torch.utils.data import Dataset

from .dataloader import DataLoaderWrapper


def calculate_mean_std(
    dataset: Dataset, batch_size: int = 64, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the mean and standard deviation of a dataset.

    Args:
        dataset (Dataset): The dataset to calculate mean and std for.
        batch_size (int): Batch size for loading data in chunks.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation for each channel.
    """

    loader = DataLoaderWrapper(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    ).get_loader()

    # Get number of channels from the first batch
    first_batch = next(iter(loader))[0]
    num_channels = first_batch.size(1)  # Number of channels

    # Initialize mean and std tensors
    mean = torch.zeros(num_channels, device=device)
    std = torch.zeros(num_channels, device=device)
    total_samples = 0

    for images, _ in loader:
        images = images.to(device)
        batch_samples = images.size(0)  # Batch size (last batch might be smaller)
        images = images.view(
            batch_samples, images.size(1), -1
        )  # Flatten each image channel
        mean += images.mean(2).sum(0)  # Sum mean of each channel
        std += images.std(2).sum(0)  # Sum std of each channel
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean.cpu(), std.cpu()  # Return to CPU for further usage
