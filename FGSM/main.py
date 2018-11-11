from model import *
from data import *
from test import *

import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['figure.figsize'] = (6.0, 6.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

path = "./examples/"

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


lenet = LeNet().to(device)
lenet.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
lenet.eval()

accuracies, adv_exs = [], []
for eps in epsilons:
    acc, ex = test(lenet, device, test_loader, eps)
    accuracies.append(acc)
    adv_exs.append(ex)
    if len(ex) != 0:
        for i in range(len(ex)):
            plt.imsave(path + "ex_" + str(eps) + "_" + str(i) + ".png", ex[i])
    print("Adversarial examples saved to " + path)
 

plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 0.35, step=0.05))
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
