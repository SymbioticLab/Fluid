#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:30:05 2020

@author: liujiachen
"""
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from ray import tune
from ray.util.sgd.utils import BATCH_SIZE
import torch

import workloads.common as com
DATA_PATH, RESULTS_PATH = com.detect_paths()





class CNN(nn.Module):
    """CNN."""

    def __init__(self, config):
        """CNN Builder."""
        super(CNN, self).__init__()

        # 64, 128, 0.05
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=int(config['conv_1']), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=int(config['conv_1']), out_channels=int(config['conv_2']), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(config['conv_2'])),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=int(config['conv_2']), out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=int(config['dropout_1'])),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        #1024, 0.1
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, int(config['dense_1'])),
            nn.ReLU(inplace=True),
            nn.Linear(int(config['dense_1']), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=int(config['dropout_2'])),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x

def model_creator(config):
    return CNN(config)

def exp_metric():
    return dict(metric="val_accuracy", mode="max")

def create_stopper():
    return com.MetricStopper(0.9, **exp_metric())

def data_creator(config):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        DATA_PATH, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True )

    testset = datasets.CIFAR10(
        DATA_PATH, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=64 , shuffle=True )

#    classes = ('plane', 'car', 'bird', 'cat', 'deer',
#               'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader

def optimizer_creator(model, config):
    """Returns optimizer"""
    return torch.optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.1),
        momentum=config.get("momentum", 0.9))


def loss_creator(config):
    return nn.CrossEntropyLoss()

def create_sample_space():
    mutations = {
            # distribution for resampling
            "lr": lambda *_ : np.random.uniform(0.001, 1),
            # allow perturbations within this set of categorical values
            "momentum": lambda *_: np.random.uniform(0.1, 0.9),
            "conv_1" : [32, 64, 128],
            "conv_2" : [ 64, 128,256],
            "dropout_1" : lambda *_: np.random.uniform(0.01, 0.1),
            "dropout_2" : lambda *_: np.random.uniform(0.05, 0.2),
            "dense_1" : [  128,256, 512, 1024],
    }
    cap_explore = com.create_cap_explore_fn(mutations, [
        ('lr', 1e-3, 1),
        ('momentum', 0.1, 0.9),
        ("conv_1", 32, 128),
        ('conv_2', 64, 256),
        ('dropout_1', 0.01, 0.1),
        ('dropout_2', 0.05, 0.2),
        ('dense_1', 128, 1024),
    ])
    return mutations, cap_explore




def create_search_space():
    mutations, cap_explore = create_sample_space()
    return {
        key: tune.sample_from(val)
        for key, val in mutations.items()
    }



def create_ch():
    import ConfigSpace as CS
    # BOHB uses ConfigSpace for their hyperparameter search space
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("lr", lower=0.001, upper=1, log=True))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("momentum", lower=0.1, upper=0.9))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("dropout_1", lower=0.01, upper=0.1))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("dropout_2", lower=0.05, upper=0.2))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("conv_1", lower=32, upper=128))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("conv_2", lower=64, upper=256))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("dense_1", lower=128, upper=1024))

    return config_space




def create_skopt_space():
    from skopt.space.space import Real

    return [
        Real(1e-3, 1, prior='log-uniform'),
        (0.1, 0.9),
        (0.01, 0.1),
        (0.05, 0.2),
        (32,128),
        (64,256),
        (128,1024),
    ], [
        'lr',
        'momentum',
        'dropout_1',
        'dropout_2',
        'conv_1',
        'conv_1',
        'dense_1',
    ]



