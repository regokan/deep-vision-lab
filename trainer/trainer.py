from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch


class Trainer(ABC):
    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, device="cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    @abstractmethod
    def forward_pass(self, inputs):
        """Define forward pass, to be implemented by subclasses."""
        pass

    @abstractmethod
    def compute_loss(self, outputs, targets):
        """Define loss computation, to be implemented by subclasses."""
        pass

    def train_one_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass and loss calculation
            outputs = self.forward_pass(inputs)
            loss = self.compute_loss(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.train_losses.append(epoch_loss)
        return epoch_loss

    def validate(self):
        """Validate the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward_pass(inputs)
                loss = self.compute_loss(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        accuracy = 100 * correct / total
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(accuracy)
        return epoch_loss, accuracy

    def train(self, num_epochs, save_path="best_model.pth"):
        """Train the model and validate after each epoch."""
        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch()
            val_loss, val_accuracy = self.validate()

            print(
                f"Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.2f}%"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Best model saved with val loss: {val_loss:.4f}")

    def plot_metrics(self):
        """Plots training and validation loss and validation accuracy over epochs."""
        epochs = range(1, len(self.train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(epochs, self.train_losses, "b-", label="Training Loss")
        ax1.plot(epochs, self.val_losses, "r-", label="Validation Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2.plot(epochs, self.val_accuracies, "g-", label="Validation Accuracy")
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()

        plt.tight_layout()
        plt.show()
