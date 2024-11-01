from typing import List, Optional

import torch
from torch.utils.data import (
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)


class SamplerWrapper:
    def __init__(
        self,
        dataset: Dataset,
        sampler_type: str = "random",
        indices: Optional[List[int]] = None,
        weights: Optional[List[float]] = None,
        replacement: bool = True,
    ):
        """
        A wrapper class for PyTorch samplers, providing an easy interface to choose between different sampling strategies.

        Args:
            dataset (Dataset): The dataset to sample from.
            sampler_type (str): Type of sampler ('random', 'sequential', 'subset_random', 'weighted_random').
            indices (Optional[List[int]]): Indices for SubsetRandomSampler.
            weights (Optional[List[float]]): Weights for WeightedRandomSampler.
            replacement (bool): Whether to sample with replacement (for WeightedRandomSampler).
        """
        self.dataset = dataset
        self.sampler_type = sampler_type
        self.indices = indices
        self.weights = weights
        self.replacement = replacement

        # Initialize the sampler
        self.sampler = self._create_sampler()

    def _create_sampler(self) -> Sampler:
        """Create and return the selected sampler."""
        if self.sampler_type == "random":
            return RandomSampler(self.dataset)
        elif self.sampler_type == "sequential":
            return SequentialSampler(self.dataset)
        elif self.sampler_type == "subset_random":
            if self.indices is None:
                raise ValueError("indices must be provided for SubsetRandomSampler")
            return SubsetRandomSampler(self.indices)
        elif self.sampler_type == "weighted_random":
            if self.weights is None:
                raise ValueError("weights must be provided for WeightedRandomSampler")
            weights_tensor = torch.tensor(self.weights, dtype=torch.double)
            return WeightedRandomSampler(
                weights=weights_tensor,
                num_samples=len(self.dataset),
                replacement=self.replacement,
            )
        else:
            raise ValueError(
                "sampler_type must be one of 'random', 'sequential', 'subset_random', or 'weighted_random'"
            )

    def get_sampler(self) -> Sampler:
        """Return the sampler instance."""
        return self.sampler

    def display_sampler_info(self) -> None:
        """Display information about the sampler configuration."""
        print(f"Sampler type: {self.sampler_type.capitalize()}")
        if self.sampler_type == "subset_random" and self.indices:
            print(f"Number of subset indices: {len(self.indices)}")
        elif self.sampler_type == "weighted_random" and self.weights:
            print(f"Weight-based sampling with replacement: {self.replacement}")
