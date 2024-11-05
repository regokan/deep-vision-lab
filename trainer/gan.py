from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from criterion import CriterionWrapper


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
        is_conv=False,
        image_shape=(28, 28),
        criterion_name="bce_with_logits",
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
        self.is_conv = is_conv
        self.image_shape = image_shape
        self.criterion_name = criterion_name

    def noisy_labels(self, labels, noise_factor=0.1):
        """
        Adds Gaussian noise to labels to introduce uncertainty for the discriminator.
        Parameters:
            labels (Tensor): The original labels (1 for real, 0 for fake).
            noise_factor (float): The amount of noise to add (0.1 is common).
        """
        return labels + noise_factor * torch.randn_like(labels)

    def add_instance_noise(self, images, noise_factor=0.1):
        """
        Adds Gaussian noise to images (real or fake) to introduce instance noise.
        Parameters:
            images (Tensor): The original images (real or fake).
            noise_factor (float): The amount of noise to add (0.1 is common).
        """
        return images + noise_factor * torch.randn_like(images)

    def real_loss(self, D_out, smooth=False, label_noise=0.0):
        """
        Computes the real loss, with optional label smoothing and label noise.
        """
        batch_size = D_out.size(0)
        labels = torch.ones(batch_size, device=self.device) * (0.9 if smooth else 1.0)

        # Add label noise if specified
        if label_noise > 0:
            labels = labels + label_noise * torch.randn_like(labels)

        criterion = CriterionWrapper(self.criterion_name).get_criterion()
        return criterion(D_out.squeeze(), labels)

    def fake_loss(self, D_out, label_noise=0.0):
        """
        Computes the fake loss with optional label noise.
        """
        batch_size = D_out.size(0)
        labels = torch.zeros(batch_size, device=self.device)

        # Add label noise if specified
        if label_noise > 0:
            labels = labels + label_noise * torch.randn_like(labels)
        criterion = CriterionWrapper(self.criterion_name).get_criterion()
        return criterion(D_out.squeeze(), labels)

    def train_one_epoch(
        self,
        print_every=100,
        smooth=False,
        g_updates=1,
        label_noise=0,
        instance_noise=0,
    ):
        """
        Train the GAN for one epoch with optional label smoothing, configurable generator update frequency,
        label noise, and instance noise.
        """
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0

        for batch_i, (real_images, _) in enumerate(self.train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)  # * 2 - 1  # Rescale to [-1, 1]

            # Add instance noise to real images
            if instance_noise > 0:
                real_images = self.add_instance_noise(
                    real_images, noise_factor=instance_noise
                )

            # ============================================
            #            TRAIN THE DISCRIMINATOR
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # ============================================
            self.d_optimizer.zero_grad()

            # Train on real images with real loss
            if self.is_conv:
                d_output_real = self.discriminator(real_images)
            else:
                d_output_real = self.discriminator(real_images.view(batch_size, -1))
            d_loss_real = self.real_loss(
                d_output_real, smooth=smooth, label_noise=label_noise
            )
            D_x = d_output_real.mean().item()  # Track real image score for monitoring

            # Generate fake images and add instance noise if needed
            z = torch.randn(batch_size, self.z_size, device=self.device)
            fake_images = self.generator(z)
            if instance_noise > 0:
                fake_images = self.add_instance_noise(
                    fake_images, noise_factor=instance_noise
                )

            d_output_fake = self.discriminator(fake_images.detach())
            d_loss_fake = self.fake_loss(d_output_fake, label_noise=label_noise)
            D_G_z1 = (
                d_output_fake.mean().item()
            )  # Track fake image score for monitoring

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.d_optimizer.step()

            # =========================================
            #            TRAIN THE GENERATOR
            # (2) Update G network: maximize log(D(G(z)))
            # =========================================
            # Train the generator multiple times
            g_loss = 0  # Reset generator loss for averaging
            for _ in range(g_updates):
                self.g_optimizer.zero_grad()

                # Generate fake images and compute generator loss
                # Also to keep track of graph on backpropagation
                z = torch.randn(batch_size, self.z_size, device=self.device)
                fake_images = self.generator(z)
                g_output = self.discriminator(fake_images)
                g_loss = self.real_loss(g_output)  # We want D(G(z)) close to 1

                # Backward pass and optimization for generator
                g_loss.backward()
                self.g_optimizer.step()

            D_G_z2 = g_output.mean().item()  # Track generator's success score
            # Accumulate generator loss for early stopping
            total_g_loss += g_loss.item()

            # Print losses and scores periodically
            if batch_i % print_every == 0:
                time = str(datetime.now()).split(".", maxsplit=1)[0]
                print(
                    f"{time} | Batch {batch_i}/{len(self.train_loader)} | "
                    f"d_loss: {d_loss.item():.4f} | g_loss: {g_loss.item():.4f} | "
                    f"D(x): {D_x:.4f} | D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
                )

            # Store the losses for later analysis
            self.losses.append((d_loss.item(), g_loss.item()))

        return total_g_loss / len(self.train_loader)

    def train(
        self,
        num_epochs,
        print_every=100,
        smooth=False,
        patience=5,
        view_samples=False,
        g_updates=1,
        label_noise=0,
        instance_noise=0,
    ):
        """Train the GAN for multiple epochs, with optional label smoothing and early stopping."""
        for _ in range(num_epochs):
            _ = self.train_one_epoch(
                print_every=print_every,
                smooth=smooth,
                g_updates=g_updates,
                label_noise=label_noise,
                instance_noise=instance_noise,
            )

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

        # Scale the images from [-1, 1] to [0, 1] if necessary
        samples = (samples + 1) / 2  # This assumes samples are in [-1, 1]

        # Reshape or permute to match image shape if necessary
        samples = (
            samples.view(-1, *self.image_shape)
            if samples.dim() == 2
            else samples.permute(0, 2, 3, 1)
        )

        fig, axes = plt.subplots(1, self.sample_size, figsize=(self.sample_size, 1))
        for img, ax in zip(samples, axes):
            ax.imshow(img.numpy())
            ax.axis("off")
        plt.show()
