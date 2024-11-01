from typing import List

import torch
import torch.nn as nn
from torchvision import models

from .block import BottleneckBlock, ResidualBlock


class ResNet(nn.Module):
    def __init__(
        self,
        version: str = "resnet18",
        num_classes: int = 1000,
        pretrained: bool = False,
    ):
        """
        Initializes ResNet architecture manually or loads pretrained ResNet from torchvision.

        Args:
            version (str): Specifies the ResNet version ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152').
            num_classes (int): Number of output classes.
            pretrained (bool): If True, loads a pretrained ResNet from torchvision.
        """
        super().__init__()
        self.in_channels = 64

        # Layer configuration for each ResNet version
        resnet_versions = {
            "resnet18": (ResidualBlock, [2, 2, 2, 2]),
            "resnet34": (ResidualBlock, [3, 4, 6, 3]),
            "resnet50": (BottleneckBlock, [3, 4, 6, 3]),
            "resnet101": (BottleneckBlock, [3, 4, 23, 3]),
            "resnet152": (BottleneckBlock, [3, 8, 36, 3]),
        }

        if version not in resnet_versions:
            raise ValueError(
                "Unsupported ResNet version. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'."
            )

        block, layers = resnet_versions[version]

        if pretrained:
            # Load pretrained model from torchvision
            self.model = getattr(models, version)(pretrained=True)
            if num_classes != 1000:
                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        else:
            # Define the ResNet architecture manually
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Create ResNet layers
            self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            # Adaptive pooling and final fully connected layer
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: Type[nn.Module], out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """
        Creates a layer consisting of `blocks` number of residual blocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "model"):
            # Use pretrained model from torchvision
            return self.model(x)
        else:
            # Custom-defined ResNet
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x
