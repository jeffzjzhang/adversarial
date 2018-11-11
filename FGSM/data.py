import torch

from torchvision import datasets, transforms

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./datasets", train=False, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])),
    batch_size=1, shuffle=True)
       
