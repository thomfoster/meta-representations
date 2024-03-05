import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, hdims=None):
        super().__init__()
        if hdims is None:
            hdims = [1000,512,256,128,56]
            
        self.flat = nn.Flatten()
        layers = []
        layers.append(nn.Linear(3 * 32 * 32, hdims[0]))
        for hdim_in, hdim_out in zip(hdims[0:-1], hdims[1:]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hdim_in, hdim_out))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hdims[-1], 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flat(x)
        return self.layers(x)