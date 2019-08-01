from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset 
import numpy as np
import torch


class CustomLabelDataset(Dataset):
    def __init__(self, path, height=28, width=28, channels=1, transform=None):
        self.data = np.load(path)['sample']
        self.labels = np.load(path)['label']
        self.height = height
        self.width = width
        self.channels = channels
        self.transform = transform

    def __getitem__(self, index):
        img_as_np = self.data[index].reshape(self.height, self.width, self.channels)
        single_image_label = self.labels[index]

        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_np)
        else:
            img_as_tensor = img_as_np

        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)

# class CustomDataset(Dataset):
#     def __init__(self, sample_path, height=28, width=28, channels=1, transform=None):
#         self.data = np.load(sample_path)
#         self.height = height
#         self.width = width
#         self.channels = channels
#         self.transform = transform

#     def __getitem__(self, index):
#         img_as_np = self.data[index].reshape(self.height, self.width, self.channels)

#         # Transform image to tensor
#         if self.transform is not None:
#             img_as_tensor = self.transform(img_as_np)
#         else:
#             img_as_tensor = img_as_np

#         # Return image and the label
#         return img_as_tensor

#     def __len__(self):
#         return len(self.data)

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)        


def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
            
    return data_loader