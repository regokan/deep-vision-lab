from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class GANTrainer:
    def __init__(
        self,
        generator,
        discriminator,
        train_loader,
        g_optimizer,
        d_optimizer,
        z_size,
        device="cpu",
        sample_size=16,
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.z_size = z_size
        self.device = device
        self.sample_size = sample_size
        self.fixed_z = torch.randn(sample_size, z_size, device=device)
        self.losses = []
        self.samples = []
        self.best_g_loss = float("inf")
        self.early_stopping_counter = 0

    def real_loss(self, D_out, smooth=False):
        batch_size = D_out.size(0)
        labels = torch.ones(batch_size, device=self.device) * (0.9 if smooth else 1.0)
        criterion = nn.BCEWithLogitsLoss()
        return criterion(D_out.squeeze(), labels)

    def fake_loss(self, D_out):
        batch_size = D_out.size(0)
        labels = torch.zeros(batch_size, device=self.device)
        criterion = nn.BCEWithLogitsLoss()
        return criterion(D_out.squeeze(), labels)

    def train_one_epoch(self, print_every=100, smooth=False):
        """Train the GAN for one epoch with optional label smoothing."""
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0

        for batch_i, (real_images, _) in enumerate(self.train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device) * 2 - 1  # Rescale to [-1, 1]

            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # ============================================
            self.d_optimizer.zero_grad()

            # Train on real images
            d_output_real = self.discriminator(real_images.view(batch_size, -1))
            d_loss_real = self.real_loss(d_output_real, smooth=smooth)

            # Train on fake images
            z = torch.randn(batch_size, self.z_size, device=self.device)
            fake_images = self.generator(z)
            d_output_fake = self.discriminator(fake_images.detach())
            d_loss_fake = self.fake_loss(d_output_fake)

            # Total discriminator loss and backward pass
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()

            # =========================================
            #            TRAIN THE GENERATOR
            # =========================================
            self.g_optimizer.zero_grad()

            # Generate fake images and compute generator loss
            z = torch.randn(batch_size, self.z_size, device=self.device)
            fake_images = self.generator(z)
            g_output = self.discriminator(fake_images)
            g_loss = self.real_loss(g_output)  # We want D(G(z)) close to 1

            # Backward pass and optimization for generator
            g_loss.backward()
            self.g_optimizer.step()

            # Accumulate generator loss for early stopping
            total_g_loss += g_loss.item()

            # Print losses periodically
            if batch_i % print_every == 0:
                time = str(datetime.now()).split(".")[0]
                print(
                    f"{time} | Batch {batch_i}/{len(self.train_loader)} | "
                    f"d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f}"
                )

            # Store the losses for plotting
            self.losses.append((d_loss.item(), g_loss.item()))

        return total_g_loss / len(self.train_loader)

    def train(
        self, num_epochs, print_every=100, smooth=False, patience=5, view_samples=False
    ):
        """Train the GAN for multiple epochs, with optional label smoothing and early stopping."""
        for epoch in range(num_epochs):
            avg_g_loss = self.train_one_epoch(print_every=print_every, smooth=smooth)

            # Early stopping check
            if avg_g_loss < self.best_g_loss:
                self.best_g_loss = avg_g_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= patience:
                break

            # Generate samples for visualization
            self.generator.eval()  # Eval mode for generating samples
            with torch.no_grad():
                samples_z = self.generator(self.fixed_z)
                self.samples.append(samples_z)
            self.generator.train()  # Back to train mode

            if view_samples:
                # Display generated samples after each epoch
                self.view_samples(-1)

    def plot_metrics(self):
        """Plots discriminator and generator loss over batches."""
        d_losses, g_losses = zip(*self.losses)
        plt.figure(figsize=(10, 5))
        plt.plot(d_losses, label="Discriminator Loss")
        plt.plot(g_losses, label="Generator Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def view_samples(self, epoch=-1, samples=[]):
        """Displays generated images from the samples list for a specific epoch."""
        if len(samples) == 0:
            samples = self.samples[epoch].cpu().detach()
        fig, axes = plt.subplots(1, self.sample_size, figsize=(self.sample_size, 1))
        for img, ax in zip(samples, axes):
            ax.imshow(img.view(28, 28), cmap="gray")
            ax.axis("off")
        plt.show()
