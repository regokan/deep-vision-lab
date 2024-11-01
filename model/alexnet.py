import torch
import torch.nn as nn
from torchvision import models


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, pretrained: bool = False):
        """
        Initializes AlexNet architecture. If `pretrained` is True, loads the pretrained
        AlexNet from torchvision models.

        Args:
            num_classes (int): Number of output classes for classification.
            pretrained (bool): If True, loads a pretrained AlexNet from torchvision.
        """
        super().__init__()

        if pretrained:
            # Load the pretrained AlexNet from torchvision
            self.model = models.alexnet(pretrained=True)
            # Update the classifier for the specified number of classes if needed
            if num_classes != 1000:
                self.model.classifier[6] = nn.Linear(4096, num_classes)
        else:
            # Define the original AlexNet architecture manually
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass for AlexNet."""
        if hasattr(self, "model"):
            # If using pretrained model from torchvision
            return self.model(x)
        else:
            # Original AlexNet definition
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
