import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


epsilons = [0, 0.05, 0.1, 0.2, 0.25, 0.3]
pretrained_model = "lenet_mnist_model.pth"
use_cuda = True

# LeNet-5 Model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        y_ = x
        y_ = F.relu(F.max_pool2d(self.conv1(y_), 2))
        y_ = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(y_)), 2))
        y_ = y_.view(-1, 320)  # fully connected strench
        y_ = F.relu(self.fc1(y_))
        y_ = F.dropout(y_, training=self.training)
        y_ = self.fc2(y_)
        return F.log_softmax(y_, dim=1)

