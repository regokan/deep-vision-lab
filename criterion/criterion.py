from typing import Optional

import torch.nn as nn


class CriterionWrapper:
    def __init__(self, criterion_name: str = "cross_entropy", **kwargs):
        """
        Initializes a loss function (criterion) for training.

        Args:
            criterion_name (str): Name of the criterion ('cross_entropy', 'mse', 'mae', etc.).
            kwargs: Additional keyword arguments specific to each criterion.
        """
        self.criterion_name = criterion_name.lower()
        self.kwargs = kwargs

        # Initialize the criterion
        self.criterion = self._create_criterion()

    def _create_criterion(self) -> nn.Module:
        """
        Creates and returns a criterion (loss function) based on the specified configuration.

        Returns:
            nn.Module: Configured criterion instance.
        """
        if self.criterion_name == "cross_entropy":
            return nn.CrossEntropyLoss(**self.kwargs)

        elif self.criterion_name == "mse":
            return nn.MSELoss(**self.kwargs)

        elif self.criterion_name == "mae":
            return nn.L1Loss(**self.kwargs)

        elif self.criterion_name == "bce":
            return nn.BCELoss(**self.kwargs)

        elif self.criterion_name == "bce_with_logits":
            return nn.BCEWithLogitsLoss(**self.kwargs)

        elif self.criterion_name == "huber":
            return nn.SmoothL1Loss(**self.kwargs)

        elif self.criterion_name == "nll":
            return nn.NLLLoss(**self.kwargs)

        else:
            raise ValueError(
                f"Unsupported criterion: {self.criterion_name}. Choose from 'cross_entropy', 'mse', 'mae', 'bce', 'bce_with_logits', 'huber', 'nll'."
            )

    def get_criterion(self) -> nn.Module:
        """
        Returns the initialized criterion (loss function).

        Returns:
            nn.Module: The criterion instance.
        """
        return self.criterion

    def display_criterion_info(self) -> None:
        """
        Prints information about the criterion configuration.
        """
        print(f"Criterion: {self.criterion_name.capitalize()}")
        for key, value in self.kwargs.items():
            print(f"{key}: {value}")
