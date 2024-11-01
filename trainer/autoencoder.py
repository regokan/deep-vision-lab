from .trainer import Trainer


class AutoencoderTrainer(Trainer):
    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, device="cpu"
    ):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device)

    def forward_pass(self, inputs):
        """Performs a forward pass for an Autoencoder."""
        return self.model(inputs)

    def compute_loss(self, outputs, targets):
        """Calculates reconstruction loss (e.g., MSE) between input and output."""
        return self.criterion(outputs, targets)  # Here, targets = inputs
