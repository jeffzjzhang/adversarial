from data import *
from model import *
from train import *
from test import *

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import argparse


plt.rcParams['figure.figsize'] = (6.0, 6.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

epsilons = [0, 0.001, 0.004, 0.008, 0.012]
accuracies = []
model_name = "resnet18"
num_classes = 10
num_epochs = 10
feature_extract = True
path = "./examples/"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
args = parser.parse_args()

print("CUDA Available: ", torch.cuda.is_available())
if str(device) == "cuda":
    print("Using device: CUDA")
else:
    print("Using device: cpu")

resnet18, input_size = initialize_resnet18(num_classes, feature_extract)
# print(resnet18)
resnet18 = resnet18.to(device)

params_to_update = []
for name, param in resnet18.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

if args.model == None:
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    resnet18_ft = train(resnet18, train_loader, optimizer, num_epochs=num_epochs)
else:
    resnet18_ft = resnet18
    resnet18_ft.load_state_dict(torch.load(args.model))

test(resnet18_ft, test_loader)

if args.model == None:
    torch.save(resnet18_ft.state_dict(), "./model_resnet18.pkl")

for eps in epsilons:
    acc, exs = attack(resnet18_ft, test_loader, eps)
    accuracies.append(acc)
    for i in range(len(exs)):
        # plt.imsave("./examples/" + str(eps) + "_" + str(i) + ".png", exs[i].reshape(input_size, input_size, -1))
        plt.imsave("./examples/" + str(eps) + "_" + str(i) + ".png", np.transpose(exs[i], (1, 2, 0)))

plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 0.012, step=0.04))
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
