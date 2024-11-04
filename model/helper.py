import torch.nn as nn


def initialize_weights(m):
    classname = m.__class__.__name__

    if isinstance(m, nn.Conv2d):
        # Initialize Conv2d layers with normal distribution (mean=0, std=0.02)
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):
        # Initialize BatchNorm2d layers (mean=1, std=0.02) and set bias to 0
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        # Initialize Linear layers with Xavier uniform distribution
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
