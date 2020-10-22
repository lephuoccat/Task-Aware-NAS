# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:57:38 2020

@author: catpl
"""

import os
import argparse

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST

# quickdraw! class object
class_object = ['apple', 'baseball-bat', 'bear', 'envelope', 'guitar', 'lollipop', 'moon', 'mouse', 'mushroom', 'rabbit']

transform = transforms.ToTensor()

class feature_Dataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i, :]
        data = np.asarray(data).astype(np.uint8).reshape(28, 28)
        
        if self.transforms:
            data = self.transforms(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
        

# load full limit-class
def full_class_dataset(dataset, limit, class_object, args):
    if (dataset == 'MNIST'):
        print("Loading full MNIST dataset...")
        data_train = MNIST(root='./data', train=True, download=True, transform=transform)
        data_test = MNIST(root='./data', train=False, download=True, transform=transform)
    elif (dataset == 'fMNIST'):
        print("Loading full Fashion-MNIST dataset...")
        data_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        data_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        print("Loading full QuickDraw! dataset...")
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(len(class_object)):
            # load npy file and concatenate data
            ob = np.load('./data/quickdraw/full_numpy_bitmap_'+ class_object[i] +'.npy')
            # choose train size and test size
            train = ob[0:5000,]
            test = ob[5000:6000,]
            train_label = np.concatenate((train_label, i * np.ones(train.shape[0])), axis=0)
            test_label = np.concatenate((test_label, i * np.ones(test.shape[0])), axis=0)
            
            if i == 0:
                train_data = train
                test_data = test
            else:
                train_data = np.concatenate((train_data, train), axis=0)
                test_data = np.concatenate((test_data, test), axis=0)
        
        # generate dataloader
        trainset = feature_Dataset(train_data, train_label, transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)
        
        testset = feature_Dataset(test_data, test_label, transform)
        testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)
                    
        
    if (dataset == 'MNIST' or dataset == 'fMNIST'):
        # train batch
        idx = (data_train.targets < limit)
        data_train.targets = data_train.targets[idx]
        data_train.data = data_train.data[idx]
        train_label = data_train.targets.cpu().detach().numpy()
        trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=False)
        # test batch
        idx = (data_test.targets < limit)
        data_test.targets = data_test.targets[idx]
        data_test.data = data_test.data[idx]
        test_label = data_test.targets.cpu().detach().numpy()
        testloader = DataLoader(data_test, batch_size=args.batch_size_train, shuffle=False)
    
    return trainloader, testloader, train_label, test_label


# load data for num-indicator
def indicator_dataset(dataset, num, limit, class_object, args):
    if (dataset == 'MNIST'):
        print("Loading {}-indicator for MNIST dataset...".format(num))
        data_train = MNIST(root='./data', train=True, download=True, transform=transform)
        data_test = MNIST(root='./data', train=False, download=True, transform=transform)
    elif (dataset == 'fMNIST'):
        print("Loading full Fashion-MNIST dataset...")
        data_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        data_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        print("Loading full QuickDraw! dataset...")
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        for i in range(len(class_object)):
            # load npy file and concatenate data
            ob = np.load('./data/quickdraw/full_numpy_bitmap_'+ class_object[i] +'.npy')
            # choose train size and test size
            train = ob[0:5000,]
            test = ob[5000:6000,]
            train_label = np.concatenate((train_label, i * np.ones(train.shape[0])), axis=0)
            test_label = np.concatenate((test_label, i * np.ones(test.shape[0])), axis=0)
            
            if i == 0:
                train_data = train
                test_data = test
            else:
                train_data = np.concatenate((train_data, train), axis=0)
                test_data = np.concatenate((test_data, test), axis=0)
        
        train_label[train_label != num] = -1
        train_label[train_label == num] = 1
        train_label[train_label == -1] = 0
        
        test_label[test_label != num] = -1
        test_label[test_label == num] = 1
        test_label[test_label == -1] = 0
        
        # generate dataloader
        trainset = feature_Dataset(train_data, train_label.astype(int), transform)
        trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)
        
        testset = feature_Dataset(test_data, test_label.astype(int), transform)
        testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)
    
    if (dataset == 'MNIST' or dataset == 'fMNIST'):
        # train batch
        idx = (data_train.targets < limit)
        data_train.targets = data_train.targets[idx]
        data_train.data = data_train.data[idx]
        data_train.targets[data_train.targets != num] = -1
        data_train.targets[data_train.targets == num] = 1
        data_train.targets[data_train.targets == -1] = 0
        train_label = data_train.targets.cpu().detach().numpy()
        trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=False)
        # test batch
        idx = (data_test.targets < limit)
        data_test.targets = data_test.targets[idx]
        data_test.data = data_test.data[idx]
        data_test.targets[data_test.targets != num] = -1
        data_test.targets[data_test.targets == num] = 1
        data_test.targets[data_test.targets == -1] = 0
        test_label = data_test.targets.cpu().detach().numpy()
        testloader = DataLoader(data_test, batch_size=args.batch_size_train, shuffle=False)
    
    return trainloader, testloader, train_label, test_label


# odd vs even tasks
def odd_even_dataset(dataset, limit, args):
    transform = transforms.ToTensor()
    if (dataset == 'MNIST'):
        print("Loading odd vs even MNIST dataset...")
        data_train = MNIST(root='./data', train=True, download=True, transform=transform)
        data_test = MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        print("Loading odd vs even Fashion-MNIST dataset...")
        data_train = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        data_test = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # train batch
    idx = (data_train.targets < limit)
    data_train.targets = data_train.targets[idx]
    data_train.data = data_train.data[idx]
    data_train.targets[(data_train.targets % 2) == 0] = -1
    data_train.targets[(data_train.targets % 2) != 0] = 0
    data_train.targets[(data_train.targets % 2) == 0] = 1
    train_label = data_train.targets.cpu().detach().numpy()
    trainloader = DataLoader(data_train, batch_size=args.batch_size_train, shuffle=False)
    # test batch
    idx = (data_test.targets < limit)
    data_test.targets = data_test.targets[idx]
    data_test.data = data_test.data[idx]
    data_test.targets[(data_test.targets % 2) == 0] = -1
    data_test.targets[(data_test.targets % 2) != 0] = 0
    data_test.targets[(data_test.targets % 2) == 0] = 1
    test_label = data_test.targets.cpu().detach().numpy()
    testloader = DataLoader(data_test, batch_size=args.batch_size_train, shuffle=False)
    
    return trainloader, testloader, train_label, test_label



# CIFAR10 indicator
def CIFAR10_indicator(num, args):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
     
    train_label = np.array(data_train.targets)
    train_label[train_label != num] = -1
    train_label[train_label == num] = 1
    train_label[train_label == -1] = 0
    data_train.targets = list(train_label)
    trainloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size_train, shuffle=True)
    
    test_label = np.array(data_test.targets)
    test_label[test_label != num] = -1
    test_label[test_label == num] = 1
    test_label[test_label == -1] = 0
    data_test.targets = list(test_label)
    # print(data_test.targets)
    testloader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size_train, shuffle=False)
    
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, train_label, test_label


