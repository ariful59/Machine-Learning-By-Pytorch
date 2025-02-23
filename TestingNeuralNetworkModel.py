import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

input_size = 784
hidden_layer = 400
num_class = 10
batch_size = 64

testing_data = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
)

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


model = NeuralNetwork(input_size, hidden_layer, num_class).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

with torch.no_grad():
    correct = 0
    num_step = len(testing_loader.dataset)
    for X, Y in testing_loader:
        X = X.reshape(-1, 784).to(device)
        Y = Y.to(device)
        output = model(X)
        correct += (output.argmax(1) == Y).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / num_step}%')