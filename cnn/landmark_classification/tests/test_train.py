import tempfile

import numpy as np
import pytest

from cnn.landmark_classification.src.data import get_data_loaders
from cnn.landmark_classification.src.model import MyModel
from cnn.landmark_classification.src.optimizer import get_loss, get_optimizer
from cnn.landmark_classification.src.train import (
    one_epoch_test,
    optimize,
    train_one_epoch,
    valid_one_epoch,
)


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    model = MyModel(50)
    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lt = train_one_epoch(data_loaders["train"], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"


def test_optimize(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
