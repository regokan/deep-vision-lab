from typing import Dict, Optional

import torch
from torch import nn, optim


class OptimizerWrapper:
    def __init__(
        self,
        model: nn.Module,
        optimizer_name: str = "adam",
        lr: float = 0.001,
        weight_decay: float = 0.0,
        momentum: Optional[float] = None,
        betas: Optional[tuple] = (0.9, 0.999),
        **kwargs,
    ):
        """
        Initializes an optimizer for a given model.

        Args:
            model (nn.Module): The model to optimize.
            optimizer_name (str): Name of the optimizer ('sgd', 'adam', 'adamw', etc.).
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 regularization).
            momentum (Optional[float]): Momentum factor (only for optimizers that support it).
            betas (Optional[tuple]): Coefficients for computing running averages of gradient and its square (for Adam-based optimizers).
            kwargs: Additional keyword arguments for specific optimizers.
        """
        self.model = model
        self.optimizer_name = optimizer_name.lower()
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.betas = betas
        self.kwargs = kwargs

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

    def _create_optimizer(self) -> optim.Optimizer:
        """
        Creates and returns an optimizer based on the specified configuration.

        Returns:
            optim.Optimizer: Configured optimizer instance.
        """
        params = self.model.parameters()

        if self.optimizer_name == "sgd":
            return optim.SGD(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum or 0,
                **self.kwargs,
            )

        elif self.optimizer_name == "adam":
            return optim.Adam(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
                **self.kwargs,
            )

        elif self.optimizer_name == "adamw":
            return optim.AdamW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.betas,
                **self.kwargs,
            )

        elif self.optimizer_name == "rmsprop":
            return optim.RMSprop(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum or 0,
                **self.kwargs,
            )

        else:
            raise ValueError(
                f"Unsupported optimizer: {self.optimizer_name}. Choose from 'sgd', 'adam', 'adamw', 'rmsprop'."
            )

    def get_optimizer(self) -> optim.Optimizer:
        """
        Returns the initialized optimizer.

        Returns:
            optim.Optimizer: The optimizer instance.
        """
        return self.optimizer

    def update_learning_rate(self, new_lr: float):
        """
        Updates the learning rate for the optimizer.

        Args:
            new_lr (float): New learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        print(f"Updated learning rate to: {new_lr}")

    def display_optimizer_info(self) -> None:
        """
        Prints information about the optimizer configuration.
        """
        print(f"Optimizer: {self.optimizer_name.capitalize()}")
        print(f"Learning Rate: {self.lr}")
        print(f"Weight Decay: {self.weight_decay}")
        if self.momentum is not None:
            print(f"Momentum: {self.momentum}")
        if self.optimizer_name in ["adam", "adamw"]:
            print(f"Betas: {self.betas}")
