import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the AlexNet Model (as defined above)
class AlexNet(nn.Module):
    def __init__(self, num_classes=10): # Note number of classes changed
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, padding=1), # Reduced kernel size
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=3, padding=1), # Reduced kernel size
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 4096), # adjusted for 32x32 input, after 3 pooling layers we get a feature of 4x4.
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.classifier(x)
        return x

# 1. Data Loading and Preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 2. Model and Optimizer
model = AlexNet(num_classes=10).to(device) # CIFAR-10 has 10 classes
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # changed optimizer from Adam to SGD as per paper

# 3. Training Loop
num_epochs = 25

for epoch in range(num_epochs):
    running_loss = 0.0
    start_time = time.time()  # Start time for the epoch
    total_batches = len(trainloader)  # Total number of batches in the epoch

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate progress
        current_batch = i + 1
        progress = (current_batch / total_batches) * 100
        elapsed_time = time.time() - start_time
        avg_time_per_batch = elapsed_time / current_batch
        remaining_time = avg_time_per_batch * (total_batches - current_batch)

        # Print progress
        if i % 20 == 19:  # Print every 20 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Batch [{current_batch}/{total_batches}], '
                  f'Progress: {progress:.2f}%, '
                  f'Loss: {running_loss / 20:.3f}, '
                  f'Elapsed: {elapsed_time:.2f}s, '
                  f'Remaining: {remaining_time:.2f}s')
            running_loss = 0.0

    print(f'Epoch [{epoch + 1}/{num_epochs}] completed in {time.time() - start_time:.2f}s')

print('Finished Training')
# 4. Testing the model
correct = 0
total = 0
with torch.no_grad():
  for data in testloader:
      images, labels = data
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))