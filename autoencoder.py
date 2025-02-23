import torchvision.datasets
import torch
from matplotlib import pyplot as plt
from torch import nn, optim

training_data  = torchvision.datasets.MNIST(
        root= './data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
)

testing_data = torchvision.datasets.MNIST(
    root= './data',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

loader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 255),
            nn.ReLU(),
            nn.Linear(255, 10),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 255),
            nn.ReLU(),
            nn.Linear(255, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10

for epoch in range(num_epochs):
    for images, labels in loader:
        images = images.view(images.size(0), -1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def visualize_results(model, dataloader):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images.view(images.size(0), -1)
    with torch.no_grad():
        outputs = model(images)

    fig, axes = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        axes[0, i].imshow(images[i].view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(outputs[i].view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

visualize_results(model, loader)








