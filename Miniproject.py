import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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

# Function for training
def train_model(model, train_loader, test_loader, optimizer, loss_fn, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
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


model1 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
model2 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)
model3 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)


# Rest of the setup (transforms, dataset, dataloader)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Loss function and optimizer
#loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.NLLLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)

# Train the model
num_epochs = 1
train_model(model1, train_loader, test_loader, optimizer1, loss_fn, num_epochs)
train_model(model2, train_loader, test_loader, optimizer2, loss_fn, num_epochs)
train_model(model3, train_loader, test_loader, optimizer3, loss_fn, num_epochs)

print("\n\nModel1: ")
print(model1)
print("accuracy: " + str(calculate_accuracy(test_loader, model1)))

print("\n\nModel2: ")
print(model2)
print("accuracy: " + str(calculate_accuracy(test_loader, model2)))

print("\n\nModel3: ")
print(model3)
print("accuracy: " + str(calculate_accuracy(test_loader, model3)))



'''
The perfomance of several numbers of hidden layers (epochs = 10):
- 2 Hidden layers: 96.71% 
- 3 Hidden layers: 96.87%
- 3 Hidden layers: 97.26% 

Sizes of hidden layers:

Different cost functions:

'''
