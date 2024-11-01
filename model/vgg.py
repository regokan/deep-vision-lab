import torch
import torch.nn as nn
from torchvision import models


class VGG(nn.Module):
    def __init__(
        self, version: str = "vgg16", num_classes: int = 1000, pretrained: bool = False
    ):
        """
        Initializes VGG architecture (VGG11, VGG13, VGG16, VGG19). If `pretrained` is True,
        loads the pretrained VGG model from torchvision.

        Args:
            version (str): Specifies the VGG version ('vgg11', 'vgg13', 'vgg16', 'vgg19').
            num_classes (int): Number of output classes for classification.
            pretrained (bool): If True, loads a pretrained VGG from torchvision.
        """
        super().__init__()

        # Dictionary to match version string with torchvision model function
        vgg_versions = {
            "vgg11": models.vgg11,
            "vgg13": models.vgg13,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
        }

        if version not in vgg_versions:
            raise ValueError(
                "Unsupported VGG version. Choose from 'vgg11', 'vgg13', 'vgg16', 'vgg19'."
            )

        if pretrained:
            # Load the pretrained VGG model from torchvision
            self.model = vgg_versions[version](pretrained=True)
            # Update the classifier for the specified number of classes if needed
            if num_classes != 1000:
                self.model.classifier[6] = nn.Linear(4096, num_classes)
        else:
            # Define the original VGG architecture manually
            self.features = self._make_layers(version)
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

    def _make_layers(self, version: str) -> nn.Sequential:
        """Define VGG features based on version configuration."""
        # Configuration for different VGG versions
        cfgs = {
            "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
            "vgg13": [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                "M",
                512,
                512,
                "M",
                512,
                512,
                "M",
            ],
            "vgg16": [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                "M",
            ],
            "vgg19": [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                512,
                "M",
            ],
        }

        layers = []
        in_channels = 3
        for x in cfgs[version]:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for VGG."""
        if hasattr(self, "model"):
            # If using pretrained model from torchvision
            return self.model(x)
        else:
            # Original VGG definition
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
