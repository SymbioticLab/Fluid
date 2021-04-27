#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:18:27 2020

@author: liujiachen
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import tune
from ray.util.sgd.utils import BATCH_SIZE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import workloads.common as com

# from ray.tune.examples.mnist_pytorch import ConvNet


DATA_PATH, RESULTS_PATH = com.detect_paths()


class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(784, int(config["hidden_layer1"]))
        self.dropout = nn.Dropout2d(float(config["drop_out"]))
        self.fc2 = nn.Linear(int(config["hidden_layer1"]), int(config["hidden_layer2"]))
        self.fc = nn.Linear(int(config["hidden_layer2"]), 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


"""
class CNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.conv1 = nn.Conv2d(1,int(config['hidden_layer1']), 3, 1)
        self.conv2 = nn.Conv2d(int(config['hidden_layer1']),64, 3, 1)
        self.dropout1 = nn.Dropout2d(float(config['drop_out']))
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, int(config['hidden_layer2']))
        self.fc2 = nn.Linear(int(config['hidden_layer2']), 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
"""


def exp_metric():
    return dict(metric="val_accuracy", mode="max")


def create_stopper():
    return com.MetricStopper(0.988, **exp_metric())


def data_creator(config):
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader = DataLoader(
        datasets.MNIST(
            DATA_PATH, train=True, download=True, transform=mnist_transforms
        ),
        batch_size=64,
        shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST(DATA_PATH, train=False, transform=mnist_transforms),
        batch_size=64,
        shuffle=True,
    )
    return train_loader, test_loader


def model_creator(config):
    return DNN(config)


def loss_creator(config):
    return nn.NLLLoss()


def optimizer_creator(model, config):
    return optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])


def create_sample_space():
    mutations = {
        # distribution for resampling
        "lr": lambda *_: 10 ** (-10 * np.random.rand()),
        "momentum": lambda *_: np.random.uniform(0.1, 0.7),
        "hidden_layer1": lambda *_: np.random.randint(32, 700),
        "hidden_layer2": lambda *_: np.random.randint(32, 256),
        "drop_out": lambda *_: np.random.uniform(0.1, 0.9),
    }
    cap_explore = com.create_cap_explore_fn(
        mutations,
        [
            ("lr", 1e-10, 1),
            ("momentum", 0.1, 0.9),
            ("drop_out", 0.1, 0.7),
            ("hidden_layer1", 32, 700),
            ("hidden_layer2", 32, 256),
        ],
    )
    return mutations, cap_explore


def create_search_space():
    mutations, cap_explore = create_sample_space()
    return {key: tune.sample_from(val) for key, val in mutations.items()}


def create_ch():
    import ConfigSpace as CS

    # BOHB uses ConfigSpace for their hyperparameter search space
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("lr", lower=1e-10, upper=1, log=True)
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("momentum", lower=0.1, upper=0.9)
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("drop_out", lower=0.1, upper=0.7)
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("hidden_layer1", lower=32, upper=700)
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("hidden_layer2", lower=32, upper=256)
    )

    return config_space


def create_skopt_space():
    from skopt.space.space import Real

    return [
        Real(1e-10, 1, prior="log-uniform"),
        (0.1, 0.9),
        (0.1, 0.7),
        (32, 700),
        (32, 256),
    ], [
        "lr",
        "momentum",
        "drop_out",
        "hidden_layer1",
        "hidden_layer2",
    ]
