import torch

from .trainer import Trainer


class VAETrainer(Trainer):
    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, device="cpu"
    ):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, device)

    def forward_pass(self, inputs):
        """Performs a forward pass for a VAE, returning reconstructed output, mu, and log_var."""
        return self.model(inputs)  # Assumes model returns (reconstructed, mu, log_var)

    def compute_loss(self, outputs, targets):
        """Calculates VAE-specific loss with reconstruction and KL divergence components."""
        reconstructed, mu, log_var = outputs
        reconstruction_loss = self.criterion(reconstructed, targets)
        kl_divergence = (
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / targets.size(0)
        )
        return reconstruction_loss + kl_divergence
