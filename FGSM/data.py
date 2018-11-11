import torch
from torchvision import datasets, transforms

import os


input_size = 224

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(input_size),
        transforms.Resize(input_size),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(input_size),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
 
# train_datasets = datasets.ImageFolder(os.path.join(data_dir, "train"), data_transforms["train"])
# test_datasets = datasets.ImageFolder(os.path.join(data_dir, "val"), data_transforms["val"])
# 
# train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=16, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1, shuffle=True)

# train_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('./datasets/', train=True, download=True, transform=data_transforms['train']),
#         batch_size=16, shuffle=True
# )
# 
# test_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10('./datasets/', train=False, download=True, transform=data_transforms['val']),
#         batch_size=1, shuffle=True
# )

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./datasets/', train=True, download=True, transform=data_transforms['train']),
        batch_size=16, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./datasets/', train=False, download=True, transform=data_transforms['val']),
        batch_size=1, shuffle=True
)

