import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transformers
from torch import no_grad
from torch.utils.data import DataLoader

from LinearRegression import criterion, optimizer
from NeuralNetwork import training_data, testing_data, total_step

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epoch = 10
batch_size = 64
learning_rate = 0.001

transformer = transformers.Compose(
    [transformers.ToTensor(),
     transformers.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),]
)

training_data = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transformer
)

testing_data = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transformer
)

training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(training_loader)

for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(training_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1000 == 0:
            print(f'Epoch {epoch + 1}/{num_epoch}.. Steps {i/total_step:.2f}.. Loss {loss.item():.3f}')


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testing_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct/total}')

torch.save(model.state_dict(), 'CNN.pth')
print("Saved PyTorch Model State to CNN.pth")

