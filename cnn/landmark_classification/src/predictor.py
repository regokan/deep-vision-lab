import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torchvision import datasets
from tqdm import tqdm

from cnn.landmark_classification.src.helpers import get_data_location


class Predictor(nn.Module):
    def __init__(self, model, class_names, mean, std):
        super().__init__()
        self.model = model.eval()
        self.class_names = class_names

        # We use nn.Sequential and not nn.Compose because the former is compatible with torch.script, while the latter isn't
        self.transforms = nn.Sequential(
            T.Resize(
                [256]
            ),  # Single int inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean.tolist(), std.tolist()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # 1. Apply transforms
            x = self.transforms(x)
            # 2. Get the logits
            x = self.model(x)
            # 3. Apply softmax across dim=1
            x = F.softmax(x, dim=1)
            return x


def predictor_test(test_dataloader, model_reloaded):
    """
    Test the predictor. Since the predictor does not operate on the same tensors
    as the non-wrapped model, we need a specific test function (can't use one_epoch_test)
    """

    folder = get_data_location()
    test_data = datasets.ImageFolder(
        os.path.join(folder, "test"), transform=T.ToTensor()
    )

    pred = []
    truth = []
    for x in tqdm(test_data, total=len(test_dataloader.dataset), leave=True, ncols=80):
        softmax = model_reloaded(x[0].unsqueeze(dim=0))
        idx = softmax.squeeze().argmax()
        pred.append(int(x[1]))
        truth.append(int(idx))

    pred = np.array(pred)
    truth = np.array(truth)

    print(f"Accuracy: {(pred == truth).sum() / pred.shape[0]}")

    return truth, pred
