from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset, Sampler


class DataLoaderWrapper:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        sampler: Optional[Sampler] = None,
        drop_last: bool = False,
    ):
        """
        A wrapper class for DataLoader that provides additional flexibility and ease of use.

        Args:
            dataset (Dataset): The dataset to load data from.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data at the start of each epoch (ignored if sampler is provided).
            num_workers (int): Number of subprocesses to use for data loading.
            sampler (Optional[Sampler]): Custom sampler to draw samples from the dataset.
            drop_last (bool): Whether to drop the last incomplete batch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = (
            shuffle if sampler is None else False
        )  # Disable shuffle if sampler is provided
        self.num_workers = num_workers
        self.sampler = sampler
        self.drop_last = drop_last

        # Initialize the DataLoader
        self.data_loader = self._create_data_loader()

    def _create_data_loader(self) -> DataLoader:
        """Create and return the DataLoader."""
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            sampler=self.sampler,
            drop_last=self.drop_last,
        )

    def get_loader(self) -> DataLoader:
        """Return the DataLoader instance."""
        return self.data_loader

    def display_loader_info(self) -> None:
        """Display information about the DataLoader configuration."""
        print(f"DataLoader for dataset: {type(self.dataset).__name__}")
        print(f"Batch size: {self.batch_size}")
        print(f"Shuffle: {self.shuffle}")
        print(f"Number of workers: {self.num_workers}")
        print(f"Sampler: {type(self.sampler).__name__ if self.sampler else 'None'}")
        print(f"Drop last: {self.drop_last}")

    def display_sample_images(
        self, num_images: int = 16, grid_size: Tuple[int, int] = (4, 4)
    ) -> None:
        """
        Displays a grid of sample images from the DataLoader.

        Args:
            num_images (int): Number of images to display.
            grid_size (tuple): Grid size for displaying images (rows, cols).
        """
        # Fetch a batch of images
        images, _ = next(iter(self.data_loader))

        # Create a grid of images
        grid_img = vutils.make_grid(
            images[:num_images], nrow=grid_size[1], padding=2, normalize=True
        )

        # Plot using imshow
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.title("Sample Images from DataLoader")
        plt.show()
