import torch.nn as nn

class SimpleCardClassifier(nn.Module):
    """
    A simple convolutional neural network for card classification.
    """
    def __init__(self, num_classes):
        """
        Initialize the model with the number of output classes.
        Args:
            num_classes (int): Number of classes for classification.
        """
        super(SimpleCardClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Convolutional layer
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling layer
            nn.Flatten(),  # Flatten the output for fully connected layers
            nn.Linear(16 * 112 * 112, 128),  # Fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(128, num_classes)  # Output layer for classification
        )

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output predictions.
        """
        return self.model(x)

