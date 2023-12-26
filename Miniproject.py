import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformations applied on each image
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

# Loading the training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Example neural network models with different configurations
def create_model(hidden_layers, hidden_size):
    layers = [nn.Flatten()]
    input_size = 784  # MNIST images are 28x28

    # Add hidden layers
    for i in range(hidden_layers):
        layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
        layers.append(nn.ReLU())

    # Output layer
    layers.append(nn.Linear(hidden_size, 10))
    layers.append(nn.LogSoftmax(dim=1))
    return nn.Sequential(*layers)

# Example usage
model_simple = create_model(hidden_layers=1, hidden_size=128)  # Simple model with 1 hidden layer
model_complex = create_model(hidden_layers=3, hidden_size=64)  # More complex model with 3 hidden layers


# Select a model and loss function
#model = model_simple
model = model_complex
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

num_epochs = 10
# Training loop with progress printout and evaluation at each epoch
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    # Evaluate after each epoch
    accuracy = calculate_accuracy(test_loader, model)
    print(f'End of Epoch {epoch+1}, Accuracy: {accuracy}%')



'''
The perfomance of several numbers of hidden layers:
- 2 Hidden layers: 96.71% (epochs = 10)
- 3 Hidden layers: 97.26% (epochs = 50)
- 10 Hidden layers: 

Sizes of hidden layers:

Different cost functions:

'''