import pytest
import torch

from cnn.landmark_classification.src.predictor import Predictor


@pytest.fixture(scope="session")
def data_loaders():
    from cnn.landmark_classification.src.data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    from cnn.landmark_classification.src.model import MyModel
    from data.helper import calculate_mean_std

    mean, std = calculate_mean_std(data_loaders["train"].dataset.dataset)
    model = MyModel(num_classes=3, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    predictor = Predictor(model, class_names=["a", "b", "c"], mean=mean, std=std)
    out = predictor(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"
    assert out.shape == torch.Size(
        [2, 3]
    ), f"Expected an output tensor of size (2, 3), got {out.shape}"
    assert torch.isclose(
        out[0].sum(), torch.Tensor([1]).squeeze()
    ), "The output of the .forward method should be a softmax vector with sum = 1"
