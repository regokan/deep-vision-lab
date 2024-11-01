import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        """
        Initializes the Squeeze-and-Excitation (SE) block.

        Args:
            in_channels (int): Number of input channels.
            reduction (int): Reduction ratio for the intermediate fully connected layer.
        """
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size is 1x1
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()

        # Squeeze: Global Average Pooling to get a 1x1 representation per channel
        out = self.global_avg_pool(x).view(batch_size, channels)

        # Excitation: Fully connected layers with a reduction and expansion
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).view(batch_size, channels, 1, 1)

        # Scale: Multiply the input by the channel-wise weights
        return x * out
