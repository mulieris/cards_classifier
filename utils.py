import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def get_data_loaders(dataset_path, batch_size, input_size=(224, 224)):
    """
    Prepare DataLoader for training and validation datasets.
    Args:
        dataset_path (str): Path to the dataset folder.
        batch_size (int): Number of samples per batch.
        input_size (tuple): Size to resize images to (default is 224x224).

    Returns:
        DataLoader: DataLoader for the training dataset.
        list: List of class names.
    """
    transform = transforms.Compose([
        transforms.Resize(input_size),  # Resize images
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, dataset.classes

def save_model(model, path):
    """
    Save the model's state dictionary.
    Args:
        model (torch.nn.Module): The model to save.
        path (str): File path to save the model.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    """
    Load a model's state dictionary.
    Args:
        model (torch.nn.Module): The model to load into.
        path (str): File path of the saved model.
        device (torch.device): Device to map the model to.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")

def plot_losses(losses, title="Training Loss", xlabel="Epoch", ylabel="Loss"):
    """
    Plot the training loss over epochs.
    Args:
        losses (list): List of loss values for each epoch.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def get_device():
    """
    Return the available device (GPU if available, otherwise CPU).
    Returns:
        torch.device: The device to use for training and inference.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

