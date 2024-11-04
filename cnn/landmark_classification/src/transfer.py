import torch.nn as nn
import torchvision
import torchvision.models as models


def get_model_transfer_learning(model_name="resnet18", n_classes=50):

    # Get the requested architecture with appropriate weights
    if hasattr(models, model_name):
        # Update the model loading to use 'weights' if available
        model_transfer = getattr(models, model_name)(weights="DEFAULT")
    else:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])
        raise ValueError(
            f"Model {model_name} is not known. List of available models: "
            f"https://pytorch.org/vision/{torchvision_major_minor}/models.html"
        )

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Get the number of features in the last layer
    num_ftrs = model_transfer.fc.in_features

    # Replace the final fully connected layer with one that has the correct output size
    model_transfer.fc = nn.Linear(num_ftrs, n_classes)

    return model_transfer
