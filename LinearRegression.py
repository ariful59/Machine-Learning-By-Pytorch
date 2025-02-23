import torch
import torch.nn as nn

X = torch.randn(8 , 1, dtype=torch.float32)
y = torch.mul(X, 2, )

print(y)
n_sample, n_feature = X.shape

print(n_feature, n_sample)
print(y.shape)

X_test = torch.tensor([[5]], dtype=torch.float32)

class LinearReg(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearReg, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

model = LinearReg(n_feature, out_features= 1)
print(f'Prediction before training: f({X_test.item()}) = {model(X_test).item():.3f}')


learning_rate = 0.01
epochs = 1000
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if(epoch % 10 == 0):
        w, b = model.parameters()
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', loss.item())

print(f'Prediction after training: f({X_test.item()}) = {model(X_test).item():.3f}')