# Define the CNN architecture
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the enhanced CNN architecture with BatchNorm and Dropout
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.5) -> None:
        super().__init__()

        # 5 Convolutional layers with BatchNorm and Dropout
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1
            ),  # Output: (32, 224, 224)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 112, 112)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # Output: (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 56, 56)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # Output: (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 28, 28)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1
            ),  # Output: (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (256, 14, 14)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1
            ),  # Output: (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (512, 7, 7)
        )

        # Fully connected layers with Dropout
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),  # Output: (batch_size, 50)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)  # Apply convolutional layers with BatchNorm and Dropout
        x = self.fc_layers(x)  # Apply fully connected layers
        return x
