import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms, models

import numpy as np
import matplotlib.pyplot as plt

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def initialize_alexnet(num_classes, feature_extract):
    alexnet_ft = models.alexnet(pretrained=True)
    set_parameter_requires_grad(alexnet_ft, feature_extract)
    num_ftrs = alexnet_ft.classifier[6].in_features
    alexnet_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    input_size = 224  # for ImageNet requirement

    return alexnet_ft, input_size
