import torch
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm

from trainer import CNNTrainer
from utils.device import get_device

from .helpers import after_subplot

device = get_device()


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one training epoch using CNNTrainer.
    """
    trainer = CNNTrainer(model, train_dataloader, None, loss, optimizer, device)
    train_loss = trainer.train_one_epoch()
    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validates at the end of one epoch using CNNTrainer.
    """
    trainer = CNNTrainer(model, None, valid_dataloader, loss, None, device)
    valid_loss, _ = trainer.validate()  # We only need validation loss here
    return valid_loss


def optimize(
    data_loaders,
    model,
    optimizer,
    loss,
    n_epochs,
    save_path,
    interactive_tracking=False,
):
    """
    Optimizes the model using CNNTrainer.
    """
    # Initialize interactive tracking, if enabled
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    logs = {}
    valid_loss_min = None

    # Learning rate scheduler that reduces learning rate on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    # Initialize CNNTrainer
    trainer = CNNTrainer(
        model, data_loaders["train"], data_loaders["valid"], loss, optimizer, device
    )

    for epoch in range(1, n_epochs + 1):
        train_loss = trainer.train_one_epoch()  # Train for one epoch
        valid_loss, _ = trainer.validate()  # Validate and get loss

        # Print training/validation statistics
        print(
            f"Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}"
        )

        # Save the model if validation loss has decreased by more than 1%
        if (
            valid_loss_min is None
            or (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

        # Update learning rate
        scheduler.step(valid_loss)

        # Log losses and current learning rate for interactive tracking
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]
            liveloss.update(logs)
            liveloss.send()


def one_epoch_test(test_dataloader, model, loss):
    """
    Tests the model on the test data using CNNTrainer.
    """
    trainer = CNNTrainer(model, None, test_dataloader, loss, None, device)
    test_loss, accuracy = (
        trainer.validate()
    )  # Reuse validate to get test loss and accuracy

    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    return test_loss
