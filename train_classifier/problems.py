import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import utils

# from convex_adversarial import Dense, DenseSequential

import numpy as np
import torch.utils.data as td
from torch.utils.data.dataset import Dataset 
import argparse
import os
import math

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# define customized dataset (using generated data)
class CustomDataset(Dataset):
    def __init__(self, path, height, width, channels, transform=None):
        self.data = np.load(path)['sample']
        self.labels = np.load(path)['label']
        self.height = height
        self.width = width
        self.channels = channels
        self.transform = transform

    def __getitem__(self, index):

        img_as_np = self.data[index].reshape(self.height, self.width, self.channels)

        single_image_label = self.labels[index]

        # transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(img_as_np)
        else:
            img_as_tensor = img_as_np

        # return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)

## MNIST and custom MNIST loader
def mnist_loaders(batch_size, path, is_shuffle=False): 
    
    mnist_train = datasets.MNIST(path, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(path, train=False, download=True, transform=transforms.ToTensor())

    train_loader = td.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(mnist_test, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
    
    return train_loader, test_loader

def custom_mnist_loaders(batch_size, train_path, test_path, is_shuffle=False):

    mnist_train = CustomDataset(train_path, 28, 28, 1, transforms.ToTensor())
    mnist_test = CustomDataset(test_path, 28, 28, 1, transforms.ToTensor())
    
    train_loader = td.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)

    # for i, (X,y) in enumerate(test_loader):

    #     # print(X.size())
    #     # print(X)
    #     print(X.size())
    #     print(X)
    #     X = X.cpu().data.numpy().transpose(0, 2, 3, 1)
    #     print(X.shape)
    #     utils.save_images(X[:25, :, :, :], [5, 5], 'test.png')

    #     raise NotImplementedError()

    return train_loader, test_loader        

## MNIST model
def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

 
## CIFAR-10 and custom CIFAR-10 loader
def cifar_loaders(batch_size, path, is_shuffle=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    cifar_train = datasets.CIFAR10(path, train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    cifar_test = datasets.CIFAR10(path, train=False, 
                    transform=transforms.Compose([transforms.ToTensor(), normalize]))

    train_loader = td.DataLoader(cifar_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(cifar_test, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
    return train_loader, test_loader

def custom_cifar_loaders(batch_size, train_path, test_path, is_shuffle=False): 
    cifar_train = CustomDataset(train_path, 32, 32, 3, transforms.ToTensor())
    cifar_test = CustomDataset(test_path, 32, 32, 3, transforms.ToTensor())

    train_loader = td.DataLoader(cifar_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = td.DataLoader(cifar_test, batch_size=batch_size, shuffle=is_shuffle, pin_memory=True)
    
    # for i, (X,y) in enumerate(test_loader):

    #     # print(X.size())
    #     # print(X)
    #     print(X.size())
    #     print(X)
    #     X = X.cpu().data.numpy().transpose(0, 2, 3, 1)
    #     print(X.shape)
    #     utils.save_images(X[:25, :, :, :], [5, 5], 'test.png')

    #     raise NotImplementedError()

    return train_loader, test_loader

## CIFAR-10 models
def cifar_model(): 
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

def cifar_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*8*8,512),
        nn.ReLU(),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512,10)
    )
    return model
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model

# def cifar_model_resnet(N = 5, factor=10): 
#     def  block(in_filters, out_filters, k, downsample): 
#         if not downsample: 
#             k_first = 3
#             skip_stride = 1
#             k_skip = 1
#         else: 
#             k_first = 4
#             skip_stride = 2
#             k_skip = 2
#         return [
#             Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)), 
#             nn.ReLU(), 
#             Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0), 
#                   None, 
#                   nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)), 
#             nn.ReLU()
#         ]

#     conv1 = [nn.Conv2d(3,16,3,stride=1,padding=1), nn.ReLU()]
#     conv2 = block(16,16*factor,3, False)
#     for _ in range(N): 
#         conv2.extend(block(16*factor,16*factor,3, False))
#     conv3 = block(16*factor,32*factor,3, True)
#     for _ in range(N-1): 
#         conv3.extend(block(32*factor,32*factor,3, False))
#     conv4 = block(32*factor,64*factor,3, True)
#     for _ in range(N-1): 
#         conv4.extend(block(64*factor,64*factor,3, False))
#     layers = (
#         conv1 + 
#         conv2 + 
#         conv3 + 
#         conv4 +
#         [Flatten(),
#         nn.Linear(64*factor*8*8,1000), 
#         nn.ReLU(), 
#         nn.Linear(1000, 10)]
#         )
#     model = DenseSequential(
#         *layers
#     )
    
#     for m in model.modules():
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#             if m.bias is not None: 
#                 m.bias.data.zero_()
#     return model


# define the argparser here for simplicity
def argparser(prefix=None, method=None, gan_type='ACGAN', 
              batch_size=50, batch_size_test=50, epochs=60, 
              verbose=200, lr=1e-3, thres=None, epsilon=0.1, seed=0,
              starting_epsilon=0.05, schedule_length=20, proj=None, 
              norm_train='l1', norm_test='l1', opt='sgd', momentum=0.9, weight_decay=5e-4): 

    parser = argparse.ArgumentParser()

    # optimizer settings
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--batch_size_test', type=int, default=batch_size_test)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)

    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=schedule_length)

    # projection settings
    parser.add_argument('--proj', type=int, default=proj)
    parser.add_argument('--norm_train', default=norm_train)
    parser.add_argument('--norm_test', default=norm_test)

    # model arguments
    parser.add_argument('--model', default=None)
    parser.add_argument('--method', type=str, default=method, 
                                choices=['baseline', 'zico_robust'])

    # task-specific arguments
    parser.add_argument('--type', default=None)
    parser.add_argument('--category', default=None)
    parser.add_argument('--tuning', default=None)

    # other arguments
    parser.add_argument('--prefix', type=str, default=prefix)
    # parser.add_argument('--ratio', type=float, default=ratio)
    # parser.add_argument('--thres', type=float, default=thres)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', default=None)

    parser.add_argument('--gan_type', type=str, default=gan_type)
    parser.add_argument('--proctitle', type=str, default="")
    

    args = parser.parse_args()
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon

    if args.prefix:
        args.proctitle += args.prefix + '/'
        if args.model is not None: 
            args.proctitle += args.model + '/' 

        args.proctitle += args.method
        if args.method != 'baseline':
            args.proctitle += '_epsilon_' + str(args.epsilon)

        if args.schedule_length >= args.epochs: 
            raise ValueError('Schedule length for epsilon ({}) is greater than '
                             'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else: 
        raise ValueErorr('Check the arugments')

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
    return args

def args2kwargs(args): 
    if args.proj is not None: 
        kwargs = {'proj' : args.proj}
    else:
        kwargs = {}
    return kwargs







# # define customized dataset with saved lipschitz constants
# class CustomLipschitzDataset(Dataset):
#     def __init__(self, path, height, width, channels, transform=None):
#         self.data = np.load(path)['sample']
#         self.labels = np.load(path)['label']
#         self.lipschitz = np.load(path)['lipschitz']
#         self.height = height
#         self.width = width
#         self.channels = channels
#         self.transform = transform

#     def __getitem__(self, index):
#         img_as_np = self.data[index].reshape(self.height, self.width, self.channels)
#         single_image_label = self.labels[index]
#         local_lipschitz = self.lipschitz[index]

#         # transform image to tensor
#         if self.transform is not None:
#             img_as_tensor = self.transform(img_as_np)
#         else:
#             img_as_tensor = img_as_np

#         # return image, label and lipschitz constant
#         return (img_as_tensor, single_image_label, local_lipschitz)

#     def __len__(self):
#         return len(self.data)