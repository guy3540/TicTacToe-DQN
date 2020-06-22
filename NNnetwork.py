import torch.nn as nn
import torch.nn.functional as F


class NNnetwork(nn.Module):
    def __init__(self):
        super(NNnetwork, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        # self.lkrelu1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(num_features=64)
        # self.lkrelu2 = nn.LeakyReLU(0.1)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        # y= self.lkrelu1(self.bn1(self.fc1(x)))
        # y= self.lkrelu2(self.bn2(self.fc2(y)))
        y = F.relu(self.bn1(self.fc1(x)))
        y = F.relu(self.bn2(self.fc2(y)))
        return self.fc3(y)