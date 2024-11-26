# Deep Vision Lab

**Deep Vision Lab** is a modular and comprehensive repository for research, code, and experimentation with deep vision models, including Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs). It utilizes [Poetry](https://python-poetry.org/) for dependency management and project configuration, simplifying the process of managing dependencies, virtual environments, and builds while ensuring reproducibility and ease of development.

---

## Getting Started

This section helps you get started with **Deep Vision Lab**, including setting up the environment, installing dependencies, and running experiments.

### Prerequisites

- Python 3.12 or later
- Poetry for dependency and environment management
- PyTorch and related libraries (installed via Poetry)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/regokan/deep-vision-lab.git
   cd deep-vision-lab
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```

3. Activate the virtual environment:
   ```bash
   poetry shell
   ```

---

## Running Experiments

The repository includes experiments and models for both GANs and CNNs. You can run the experiments by navigating to their respective folders and using Jupyter notebooks.

### Running Jupyter Notebooks

1. Install Jupyter Notebook support (if not already installed):

   ```bash
   poetry install
   ```

2. Start the notebook server:

   ```bash
   poetry run python -m notebook
   ```

3. Open the desired notebook, such as:
   - `gan/cifar/DCGAN Image Generation.ipynb`
   - `cnn/landmark_classification/cnn_from_scratch.ipynb`

---

## Project Highlights

- **Modularity**: The repository is structured to allow for easy extension and reuse of components across projects.
- **Interactive Notebooks**: Jupyter notebooks provide an intuitive interface for running and understanding experiments.
- **Scalable Infrastructure**: Terraform scripts in the `infra/` folder enable deployment on AWS EC2 instances for large-scale training.

---

## `gan/`

The `gan/` directory focuses on generative adversarial networks (GANs), offering code, notebooks, and assets for exploring GAN architectures and their applications. It is modular and includes datasets, checkpoints, and visualizations to facilitate GAN research and experimentation.

---

### Directory Structure

```
gan
├── assets
├── cifar
├── mnist
└── stylegan
```

---

### Components

---

#### **`assets/`**

Contains visual resources, such as architecture diagrams, plots, and data representations, to support understanding of GAN principles and methods.

---

#### **`cifar/`**

- Focuses on CIFAR-10 dataset experiments using GANs.
- Includes:
  - **`DCGAN Image Generation.ipynb`**: Notebook implementing a Deep Convolutional GAN (DCGAN) for generating CIFAR-10 images.
  - **`data/`**: Contains preprocessed CIFAR-10 data for quick experimentation.

---

#### **`mnist/`**

- Focuses on digit generation using the MNIST dataset.
- Includes:
  - **`GAN Image Generation.ipynb`**: Notebook showcasing GAN training and image generation on MNIST.
  - **`data/`**: Preprocessed MNIST dataset.

---

#### **`stylegan/`**

- Explores StyleGAN, a state-of-the-art GAN model for high-quality image generation.
- Includes:
  - **`Face Generation.ipynb`**: Demonstrates training StyleGAN for face generation using a small CelebA dataset.
  - **`Overview.ipynb`**: Explains the StyleGAN architecture, training process, and design innovations.

---

### How to Use

1. **Run GAN Experiments**:

   - Open the notebooks in `cifar/`, `mnist/`, or `stylegan/` using Jupyter or your preferred notebook environment.
   - Execute the cells to train the GANs or generate images using pre-trained models.

2. **Visualize GAN Components**:

   - Refer to the visual aids in the `assets/` folder to understand architecture and pipeline design.

3. **Dataset Preparation**:
   - Use the preprocessed datasets in `cifar/data/`, `mnist/data/`, or `stylegan/processed_celeba_small/` for experiments, or replace with your own datasets.

The `gan/` directory serves as a comprehensive resource for learning and experimenting with GANs, covering fundamental models like DCGAN to advanced architectures like StyleGAN.

---

## CNN Projects

This section of the repository focuses on implementing Convolutional Neural Networks (CNNs) for various classification tasks. Below are the two main projects included:

### 1. Landmark Classification

In the landmark classification project, we develop a classifier to identify various landmarks using CNNs. The project includes the following notebooks:

- **`cnn_from_scratch.ipynb`**: Demonstrates how to build a CNN from scratch, covering data loading, model definition, training, and evaluation.
- **`transfer_learning.ipynb`**: Showcases how to leverage pre-trained models for landmark classification, enabling faster training and improved performance.

#### Key Features

- **Data Loading**: Efficient setup for handling training and validation datasets.
- **Model Definition**: Custom CNN architectures designed for the task.
- **Training and Evaluation**: Includes code for training models and assessing their performance.

---

### 2. MNIST Classification

This project focuses on digit recognition using both CNNs and Multi-Layer Perceptrons (MLPs). The goal is to compare their performance on the MNIST dataset.

- **`classification.ipynb`**: Implements a CNN for handwritten digit classification, covering preprocessing, training, and evaluation.
- **`mlp/classification.ipynb`**: Implements an MLP for the same task, allowing for a direct comparison with the CNN model.

#### Key Features

- **Data Handling**: Utilities for loading and preprocessing the MNIST dataset.
- **Model Architecture**: Includes both CNN and MLP designs to highlight their differences.
- **Training and Metrics**: Detailed training processes and evaluation metrics.

The CNN projects provide a comprehensive walkthrough for building and training CNNs for image classification tasks. Whether you’re exploring CNNs or comparing their performance to MLPs, these projects serve as practical examples for learning and implementation in Python using PyTorch. This consolidates the information in your main README and keeps it accessible alongside the other repository components.

---

## `model/`

The `model` directory is a key part of the repository, housing various deep learning models and their architectural components. It is modularized to provide flexibility and reusability for a wide range of convolutional neural network (CNN) architectures and custom blocks. Below is an overview of the components:

### Structure:

```
model
├── alexnet.py
├── block
│   ├── __init__.py
│   ├── bottleneck.py
│   ├── gap.py
│   ├── residual.py
│   └── se.py
├── helper.py
├── resnet.py
└── vgg.py
```

### Components:

#### **Model Architectures**

- **`alexnet.py`**: Implements the AlexNet architecture, a pioneering CNN known for its role in the 2012 ImageNet competition.
- **`resnet.py`**: Contains the ResNet (Residual Network) implementation, which introduced skip connections to tackle vanishing gradient issues.
- **`vgg.py`**: Includes the VGG (Visual Geometry Group) network implementation, known for its simplicity and use of small filters.

#### **Building Blocks (Modular Layers)**

- **`block/`**: A collection of modular building blocks that can be integrated into different CNN architectures:
  - **`bottleneck.py`**: Implements the bottleneck block, a key component in ResNet for efficient computation.
  - **`gap.py`**: Implements Global Average Pooling (GAP), often used to reduce dimensions in feature maps.
  - **`residual.py`**: Defines residual blocks, essential for ResNet architectures.
  - **`se.py`**: Implements Squeeze-and-Excitation (SE) blocks to recalibrate channel-wise feature responses.

#### **Helper Functions**

- **`helper.py`**: Provides utility functions for common tasks such as weight initialization, layer configuration, and visualization of model components.

The `model` folder is designed to make it easy to extend existing architectures or design new ones by leveraging reusable components. Each file is well-documented, making it straightforward for researchers and developers to understand and adapt the code.

### `criterion/`

Contains modules defining loss functions and evaluation metrics for training deep vision models.

- **`criterion.py`**: Implements loss functions like cross-entropy, mean squared error, or custom losses tailored to specific tasks.
- **`__init__.py`**: Ensures the folder can be imported as a Python package.

---

### `data/`

A modularized folder to handle dataset preparation, loading, and augmentation.

- **`dataset.py`**: Defines dataset classes for various tasks, enabling efficient data handling and preprocessing.
- **`dataloader.py`**: Implements PyTorch-compatible data loaders with batching and multi-threaded data fetching.
- **`helper.py`**: Provides helper functions for tasks like data transformation, normalization, and splitting datasets.
- **`sampler.py`**: Custom sampling strategies for imbalanced datasets or task-specific requirements.

---

### `infra/`

Houses infrastructure-as-code (IaC) scripts for deploying and managing cloud-based resources for training models.

- **Terraform Scripts**:
  - **`config.tf`, `data.tf`, `networking.tf`**: Define cloud infrastructure configurations for datasets, storage, and networking.
  - **`apply.local`, `plan.local`, `destroy.local`**: Command scripts for deploying, planning, or tearing down the infrastructure.
  - **`variables.tf`**: Parameter definitions to ensure reusability and consistency.
  - **Modules**: Modular Terraform components for scalable infrastructure management.

---

### `optimize/`

Focuses on optimization strategies for training models.

- **`optimizer.py`**: Implements optimization algorithms like SGD, Adam, and RMSProp, along with learning rate schedulers.
- **`__init__.py`**: Ensures compatibility as a Python package.

---

### `trainer/`

Manages the training pipelines for various model types.

- **`trainer.py`**: A general-purpose training script that integrates models, datasets, optimizers, and evaluation metrics.
- Specialized Trainers:
  - **`cnn.py`**: Trainer for convolutional neural networks.
  - **`gan.py`**: Trainer for generative adversarial networks.
  - **`autoencoder.py`**: Trainer for autoencoders.
  - **`vae.py`**: Trainer for variational autoencoders.
- **`__init__.py`**: Sets up the folder as a package.

---

### `utils/`

Contains utility scripts to simplify repetitive tasks.

- **`device.py`**: Handles device management (e.g., GPU/CPU selection) for efficient computation.
- **`__init__.py`**: Ensures the folder is importable.

---

### Configuration Files

- **`poetry.lock`, `pyproject.toml`, `poetry.toml`**: Define dependencies and environment configurations using Poetry for streamlined package management.

---

### `Makefile`

### Updated `Makefile` Description

The `Makefile` provides automation for setting up the environment, installing dependencies, formatting code, running tests, and preparing Ubuntu-based AWS EC2 instances (e.g., `p3.2xlarge`) for training and inference jobs. Below is an explanation of the key commands:

- **`install`**: Installs all project dependencies using Poetry.

  ```bash
  poetry install
  ```

- **`format`**: Formats Python code using `isort` (for import sorting) and `black` (for consistent code styling).

  ```bash
  poetry run python -m isort .
  poetry run python -m black .
  ```

- **`test`**: Runs test cases with `pytest`, setting the `PYTHONPATH` to the project root for proper module imports.

  ```bash
  PYTHONPATH=./ poetry run python -m pytest -vv
  ```

- **`setup-ubuntu`**: Prepares an Ubuntu system (e.g., AWS EC2) for running training or inference jobs. This includes:
  - Updating the system and installing essential packages (`curl`, `python3-pip`, `python3-venv`).
  - Installing Poetry for dependency management.
  - Configuring environment variables for Poetry.
  - Resizing the storage volume for the EC2 instance to ensure sufficient space.

---

We welcome contributions, suggestions for new features, and discussions on potential improvements. If you encounter any issues or have ideas to enhance the project, feel free to open an issue or submit a pull request. Happy exploring and experimenting with **Deep Vision Lab**!
