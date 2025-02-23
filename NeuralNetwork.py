import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

training_data = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
)

testing_data = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
)

input_size = 784
hidden_layer = 400
num_class = 10
num_epoch = 2
batch_size = 64
learning_rate = 0.001


training_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layer, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layer)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = NeuralNetwork(input_size, hidden_layer, num_class)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(training_loader)
for epoch in range(num_epoch):
    for i, (X, Y) in enumerate(training_loader):
        X = X.reshape(-1, 784).to(device)
        Y = Y.to(device)

        output = model(X)
        loss = loss_fn(output, Y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 64 == 0:
            print(f'Epoch {epoch/num_epoch + 1} , steps {i+1 / total_step} , loss {loss.item()}')

with torch.no_grad():
    correct = 0
    num_step = len(testing_loader.dataset)
    for X, Y in testing_loader:
        X = X.reshape(-1, 784).to(device)
        Y = Y.to(device)
        output = model(X)
        correct += (output.argmax(1) == Y).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / num_step}%')

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


