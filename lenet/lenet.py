import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from multiprocessing import freeze_support

# Define the LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.pool = nn.AvgPool2d(2, 2)  # 2x2 kernel, stride 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 16 channels, 5x5 image size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes

    # def forward(self, x):
    #     x = self.pool(torch.relu(self.conv1(x)))
    #     x = self.pool(torch.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor
    #     x = torch.relu(self.fc1(x))
    #     x = torch.relu(self.fc2(x))
    #     x = self.fc3(x)  # No activation before softmax (handled by CrossEntropyLoss)
    #     return x
    
    def forward(self, x):
        print("Input shape:", x.shape)  # Shape at the beginning

        x = self.pool(torch.relu(self.conv1(x)))
        print("Shape after conv1 and pool:", x.shape)

        x = self.pool(torch.relu(self.conv2(x)))
        print("Shape after conv2 and pool:", x.shape)  # Check shape here

        x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor
        print("Shape after view:", x.shape) #check the shape after flattening

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    freeze_support()
    # Create an instance of the model
    model = LeNet5()

    # --- Data Loading and Preprocessing (using MNIST) ---

    # Transformations: Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Download and load training data
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=2)

    # Download and load testing data
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=1000,
                                            shuffle=False, num_workers=2)

    # --- Loss Function and Optimizer ---

    criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax and NLLLoss
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # --- Training Loop ---

    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()
            if i % 200 == 199:  # Print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # --- Testing ---

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # --- Save the trained model ---
    PATH = './lenet5_mnist.pth'
    torch.save(model.state_dict(), PATH)