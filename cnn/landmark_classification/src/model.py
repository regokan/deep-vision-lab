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


class MyEnhancedModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1
            ),  # Output: (32, 224, 224)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 112, 112)
            nn.Dropout(p=dropout),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # Output: (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 56, 56)
            nn.Dropout(p=dropout),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # Output: (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 28, 28)
            nn.Dropout(p=dropout),
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1
            ),  # Output: (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (256, 14, 14)
            nn.Dropout(p=dropout),
            nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1
            ),  # Output: (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (512, 7, 7)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),  # Output: (batch_size, 50)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class MyLeakyModel(nn.Module):
    def __init__(
        self, num_classes: int = 50, dropout: float = 0.5, negative_slope: float = 0.01
    ) -> None:
        super().__init__()

        # 5 Convolutional layers with BatchNorm and Dropout
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1
            ),  # Output: (32, 224, 224)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 112, 112)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # Output: (64, 112, 112)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 56, 56)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # Output: (128, 56, 56)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 28, 28)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1
            ),  # Output: (256, 28, 28)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (256, 14, 14)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1
            ),  # Output: (512, 14, 14)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (512, 7, 7)
        )

        # Fully connected layers with Dropout
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),  # Output: (batch_size, 50)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class MyLeakyDeeperModel(nn.Module):
    def __init__(
        self, num_classes: int = 50, dropout: float = 0.5, negative_slope: float = 0.01
    ) -> None:
        super().__init__()

        # Enhanced Convolutional layers with additional blocks
        self.conv_layers = nn.Sequential(
            # Input: (batch_size, 3, 224, 224)
            nn.Conv2d(
                3, 32, kernel_size=3, stride=1, padding=1
            ),  # Output: (batch_size, 32, 224, 224)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 32, 112, 112)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                32, 64, kernel_size=3, stride=1, padding=1
            ),  # Output: (batch_size, 64, 112, 112)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 64, 56, 56)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                64, 128, kernel_size=3, stride=1, padding=1
            ),  # Output: (batch_size, 128, 56, 56)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 128, 28, 28)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                128, 256, kernel_size=3, stride=1, padding=1
            ),  # Output: (batch_size, 256, 28, 28)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 256, 14, 14)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                256, 512, kernel_size=3, stride=1, padding=1
            ),  # Output: (batch_size, 512, 14, 14)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 512, 7, 7)
            nn.Dropout(p=dropout / 2),
            # Additional convolutional layers
            nn.Conv2d(
                512, 512, kernel_size=3, stride=1, padding=1
            ),  # Output: (batch_size, 512, 7, 7)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 512, 3, 3)
            nn.Dropout(p=dropout / 2),
            nn.Conv2d(
                512, 1024, kernel_size=3, stride=1, padding=1
            ),  # Output: (batch_size, 1024, 3, 3)
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (batch_size, 1024, 1, 1)
            nn.Dropout(p=dropout / 2),
        )

        # Fully connected layers with additional depth
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Flattened output: (batch_size, 1024 * 1 * 1) = (batch_size, 1024)
            nn.Linear(
                1024, 2048
            ),  # Input: (batch_size, 1024), Output: (batch_size, 2048)
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 1024),  # Output: (batch_size, 1024)
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),  # Output: (batch_size, 512)
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),  # Output: (batch_size, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
