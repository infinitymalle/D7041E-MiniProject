import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

# Transformations applied on each image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Loading the training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Function to create a model
def create_model(hidden_layers, hidden_size):
    layers = OrderedDict([
        ('flatten', nn.Flatten())
    ])

    # Add hidden layers
    for i in range(hidden_layers):
        layer_key = f'dense{i+1}'
        act_key = f'act{i+1}'
        in_features = 784 if i == 0 else hidden_size
        layers[layer_key] = nn.Linear(in_features, hidden_size)
        layers[act_key] = nn.ReLU()

    # Output layer
    layers['output'] = nn.Linear(hidden_size, 10)
    layers['outact'] = nn.LogSoftmax(dim=1)
    return nn.Sequential(layers)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Function to train and evaluate a model
def train_and_evaluate_model(model, loss_fn, optimizer, train_loader, test_loader, num_epochs=5):
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accuracy = calculate_accuracy(test_loader, model)
        print(f'Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy}%')

# Example usage: Testing different configurations
model_configs = [
    {'hidden_layers': 1, 'hidden_size': 128},
    {'hidden_layers': 2, 'hidden_size': 64},
    # Add more configurations as needed
]

for config in model_configs:
    model = create_model(**config)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f'Training model with {config["hidden_layers"]} hidden layers of size {config["hidden_size"]}')
    train_and_evaluate_model(model, loss_fn, optimizer, train_loader, test_loader)

# Save and load model example
torch.save(model.state_dict(), "my_model.pth")
model.load_state_dict(torch.load("my_model.pth"))
