import torch
import torch.nn as nn


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        """
        Initializes the Global Average Pooling (GAP) block.
        GAP reduces each feature map to a single value by averaging all spatial locations.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass for the GAP block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels)
        """
        return x.mean(dim=[-2, -1])  # Average over height and width
