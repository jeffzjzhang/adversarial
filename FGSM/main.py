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

epsilons = [0, 0.05, 0.1, 0.2, 0.25, 0.3]
accuracies = []
model_name = "vgg11_bn"
num_classes = 15
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

vgg11_bn, input_size = initialize_vgg11_bn(num_classes, feature_extract)
# print(vgg11_bn)
vgg11_bn = vgg11_bn.to(device)

params_to_update = []
for name, param in vgg11_bn.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

if args.model == None:
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    vgg11_bn_ft = train(vgg11_bn, train_loader, optimizer, num_epochs=num_epochs)
else:
    vgg11_bn_ft = vgg11_bn
    vgg11_bn_ft.load_state_dict(torch.load(args.model))

# test(vgg11_bn_ft, test_loader)

if args.model == None:
    torch.save(vgg11_bn_ft.state_dict(), "./model_vgg11_bn.pkl")

for eps in epsilons:
    acc, exs = attack(vgg11_bn_ft, test_loader, eps)
    accuracies.append(acc)
    for i in range(len(exs)):
        # plt.imsave("./examples/" + str(eps) + "_" + str(i) + ".png", exs[i].reshape(input_size, input_size, -1))
        plt.imsave("./examples/" + str(eps) + "_" + str(i) + ".png", np.transpose(exs[i], (1, 2, 0)))

plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 0.35, step=0.05))
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
