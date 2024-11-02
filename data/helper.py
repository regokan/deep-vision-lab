import os
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from .dataloader import DataLoaderWrapper


def calculate_mean_std(
    dataset: Dataset,
    batch_size: int = 64,
    device: str = "cpu",
    cache_file: Optional[str] = "mean_and_std.pt.local",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the mean and standard deviation of a dataset, optionally caching the result.

    Args:
        dataset (Dataset): The dataset to calculate mean and std for.
        batch_size (int): Batch size for loading data in chunks.
        device (str): Device to perform calculations on ('cpu' or 'cuda').
        cache_file (Optional[str]): Path to a cache file. If provided and the file exists,
                                    the mean and std will be loaded from the file. If not,
                                    they will be calculated and saved to this file.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation for each channel.
    """
    # Check if cached values are available
    if cache_file and os.path.exists(cache_file):
        print(f"Loading mean and std from cache: {cache_file}")
        cached_data = torch.load(cache_file, weights_only=True)
        return cached_data["mean"], cached_data["std"]

    # Initialize DataLoader for the dataset
    loader = DataLoaderWrapper(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    ).get_loader()

    # Determine number of channels from the first batch
    first_batch = next(iter(loader))[0]
    num_channels = first_batch.size(1)

    # Initialize mean and std tensors
    mean = torch.zeros(num_channels, device=device)
    std = torch.zeros(num_channels, device=device)
    total_samples = 0

    for images, _ in loader:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # Flatten each channel

        # Accumulate sum of means and stds across batches
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    # Compute the final mean and std by dividing by total samples
    mean /= total_samples
    std /= total_samples

    # Save to cache file if provided
    if cache_file:
        print(f"Saving mean and std to cache: {cache_file}")
        torch.save({"mean": mean.cpu(), "std": std.cpu()}, cache_file)

    return mean.cpu(), std.cpu()
