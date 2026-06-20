"""
train.py

Standalone training script for the aerial scene classifier.

This script trains a ResNet18 transfer learning model on the EuroSAT dataset and saves
the trained model weights to model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

# Path to the EuroSAT RGB dataset
DATA_DIR = Path("data/EuroSAT")

# Number of terrain categories
NUM_CLASSES = 10

# Number of complete passes through the training data
NUM_ROUNDS = 5

# Number of images processed at a time
BATCH_SIZE = 32

# How large each weight adjustment step is during training
LEARNING_RATE = 0.001

# -------------------------------------------------------
# Device Setup
# -------------------------------------------------------

# Use a GPU if one is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# -------------------------------------------------------
# Image Transforms
# -------------------------------------------------------

# Resize, convert, and normalize images to match what ResNet18 expects
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------------
# Dataset Loading and Splitting
# -------------------------------------------------------

# Load the full dataset from the folder structure
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Store the class names for reference
class_names = full_dataset.classes

# Calculate how many images go into each subset
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)

# The test set gets whatever is left after training and validation
test_size = total_size - train_size - val_size

# Randomly split the dataset into training, validation, and test subsets
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# Wrap each subset in a DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total images: {total_size}")
print(f"Training images: {train_size}")
print(f"Validation images: {val_size}")
print(f"Test images: {test_size}")

# -------------------------------------------------------
# Model Setup
# -------------------------------------------------------

# Load ResNet18 with pretrained ImageNet weights
model = models.resnet18(weights="IMAGENET1K_V1")

# Freeze all existing layers so their weights do not change during training
for param in model.parameters():
    param.requires_grad = False

# Find out how many input features the final layer expects
num_features = model.fc.in_features

# Replace the final layer with a new one that outputs NUM_CLASSES classes
model.fc = nn.Linear(num_features, NUM_CLASSES)

# Move the model to the selected device
model = model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

print("Model ready for training.")

# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------

for round_num in range(NUM_ROUNDS):

    # Switch the model to training mode
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Loop through the training data one batch at a time
    for images, labels in train_loader:

        # Move images and labels to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        # Clear the gradients left over from the previous batch
        optimizer.zero_grad()

        # Forward pass: feed the images through the model and get predictions
        outputs = model(images)

        # Calculate how wrong the predictions were
        loss = criterion(outputs, labels)

        # Backward pass: calculate how much each weight contributed to the loss
        loss.backward()

        # Update the model weights to reduce the loss
        optimizer.step()

        # Track running loss and accuracy
        running_loss += loss.item()
        scores, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct_predictions = (predicted == labels)
        correct += correct_predictions.sum().item()

    train_accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # Switch the model to evaluation mode for validation
    model.eval()
    val_correct = 0
    val_total = 0

    # Disable gradient tracking during validation
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            scores, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            correct_predictions = (predicted == labels)
            val_correct += correct_predictions.sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(f"Round {round_num + 1}/{NUM_ROUNDS} | Loss: {avg_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Val Accuracy: {val_accuracy:.2f}%")

print("Training complete.")

# -------------------------------------------------------
# Save the Model
# -------------------------------------------------------

# Save the trained model weights to disk
# The Streamlit app will load these weights to make predictions
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")

# -------------------------------------------------------
# Test Set Evaluation
# -------------------------------------------------------

# Evaluate the model on the test set
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        scores, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        correct_predictions = (predicted == labels)
        test_correct += correct_predictions.sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")