import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.7) -> None:
        super().__init__()

        # 5 Convolutional layers with increasing filters
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1
            ),  # Input: (3, 224, 224), Output: (32, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 112, 112)
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # Output: (64, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 56, 56)
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # Output: (128, 56, 56)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 28, 28)
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1
            ),  # Output: (256, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (256, 14, 14)
            nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1
            ),  # Output: (512, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (512, 7, 7)
        )

        # Fully connected layers with Dropout
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),  # Output: (batch_size, 50)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)  # Apply convolutional layers
        x = self.fc_layers(x)  # Apply fully connected layers
        return x
