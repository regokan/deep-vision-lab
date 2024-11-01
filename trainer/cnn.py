from .trainer import Trainer


class CNNTrainer(Trainer):
    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, device="cpu"
    ):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device)

    def forward_pass(self, inputs):
        """Performs a forward pass for a supervised CNN."""
        return self.model(inputs)

    def compute_loss(self, outputs, targets):
        """Calculates loss using the criterion with ground-truth labels."""
        return self.criterion(outputs, targets)
