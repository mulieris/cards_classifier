# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Hyperparameters and device configuration
BATCH_SIZE = 32  # Number of samples per batch
LEARNING_RATE = 0.001  # Learning rate for the optimizer
EPOCHS = 10  # Number of training epochs
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
DATASET_PATH = "./dataset/train"  # Path to the training dataset

# Define transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values to [-1, 1]
])

# Load the dataset
train_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Print dataset information
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Classes: {train_dataset.classes}")

# Define the model
class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCardClassifier, self).__init__()
        # Define a simple CNN architecture
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Convolutional layer
            nn.ReLU(),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Pooling layer
            nn.Flatten(),  # Flatten output for fully connected layers
            nn.Linear(16 * 112 * 112, 128),  # Fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(128, num_classes)  # Output layer for classification
        )

    def forward(self, x):
        return self.model(x)

# Initialize the model, loss function, and optimizer
model = SimpleCardClassifier(num_classes=len(train_dataset.classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()  # Loss function for classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Track cumulative loss for the epoch

    for images, labels in tqdm(train_loader, desc="Training"):  # Iterate through batches
        images, labels = images.to(device), labels.to(device)  # Move data to the appropriate device

        optimizer.zero_grad()  # Clear gradients from the previous step

        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        running_loss += loss.item()  # Accumulate loss

    return running_loss / len(train_loader)  # Return average loss for the epoch

if __name__ == "__main__":
    # Train the model over multiple epochs
    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss:.4f}")

    # Save the trained model
    if not os.path.exists("./models"):
        os.makedirs("./models")
    torch.save(model.state_dict(), "./models/card_classifier.pth")
    print("Model saved to ./models/card_classifier.pth")

# Track and visualize training losses
losses = []

for epoch in range(EPOCHS):
    loss = train(model, train_loader, criterion, optimizer, DEVICE)
    losses.append(loss)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss:.4f}")

# Plot training loss over epochs
plt.plot(range(1, EPOCHS + 1), losses, marker='o')
plt.xlabel("Epoch")  # Label for the x-axis
plt.ylabel("Loss")  # Label for the y-axis
plt.title("Training Loss")  # Title of the plot
plt.show()
