import torch
import torch.nn as nn
import torch.optim

from criterion import CriterionWrapper
from optimize import OptimizerWrapper
from utils.device import get_device

device = get_device()


def get_loss(device: str = device):
    """
    Get an instance of the CrossEntropyLoss (useful for classification)

    :param device: the device to move the loss to, default to best available
    """
    # Select CrossEntropyLoss, appropriate for classification tasks
    loss = CriterionWrapper(criterion_name="cross_entropy").get_criterion().to(device)
    return loss


def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0,
):
    """
    Returns an optimizer instance

    :param model: the model to optimize
    :param optimizer: one of 'SGD' or 'Adam'
    :param learning_rate: the learning rate
    :param momentum: the momentum (if the optimizer uses it)
    :param weight_decay: regularization coefficient
    :param device: the device to move the optimizer to, default to best available
    """
    if optimizer.lower() == "sgd":
        # Create an instance of the SGD optimizer
        opt = OptimizerWrapper(
            model,
            optimizer_name="SGD",
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )

    elif optimizer.lower() == "adam":
        # Create an instance of the Adam optimizer
        opt = OptimizerWrapper(
            model, optimizer_name="Adam", lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt.get_optimizer()
