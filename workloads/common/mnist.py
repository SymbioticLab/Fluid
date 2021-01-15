import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from ray import tune
from ray.util.sgd.utils import BATCH_SIZE
from ray.tune.examples.mnist_pytorch import ConvNet

import workloads.common as com


DATA_PATH, RESULTS_PATH = com.detect_paths()


def exp_metric():
    return dict(metric="val_accuracy", mode="max")


def create_stopper():
    return com.MetricStopper(0.975, **exp_metric())


def data_creator(config):
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    train_loader = DataLoader(
        datasets.MNIST(
            DATA_PATH,
            train=True,
            download=True,
            transform=mnist_transforms),
        batch_size=64,
        shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST(DATA_PATH, train=False, transform=mnist_transforms),
        batch_size=64 ,
        shuffle=True)
    return train_loader, test_loader


def model_creator(config):
    return ConvNet()


def loss_creator(config):
    return nn.NLLLoss()


def optimizer_creator(model, config):
    return optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )


def create_sample_space():
    mutations = {
        # distribution for resampling
        "lr": lambda *_: 10**(-10 * np.random.rand()),
        "momentum": lambda *_: np.random.uniform(0.1, 0.9),
    }
    cap_explore = com.create_cap_explore_fn(mutations, [
        ('lr', 1e-10, 1),
        ('momentum', 0.1, 0.9),
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
        CS.UniformFloatHyperparameter("lr", lower=1e-10, upper=1, log=True))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("momentum", lower=0.1, upper=0.9))

    return config_space


def create_skopt_space():
    from skopt.space.space import Real

    return [
        Real(1e-10, 1, prior='log-uniform'),
        (0.1, 0.9),
    ], [
        'lr',
        'momentum',
    ]
