#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:49:12 2020

@author: liujiachen
"""
import warnings

import matplotlib.style as style
import ray
import tensorflow as tf
import torch
import torch.optim as optim
from hyperopt import hp
from ray import tune
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, test, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from torchvision import datasets

try:
    tf.get_logger().setLevel("INFO")
except Exception as exc:
    print(exc)
warnings.simplefilter("ignore")
style.use("ggplot")
datasets.MNIST("~/data", train=True, download=True)


def train_mnist(config):
    model = ConvNet()
    train_loader, test_loader = get_data_loaders()

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    for i in range(20):
        train(model, optimizer, train_loader)  # Train for 1 epoch
        acc = test(model, test_loader)  # Obtain validation accuracy.
        tune.track.log(mean_accuracy=acc)  # here
        if i % 5 == 0:
            torch.save(
                model, "./model.pth"
            )  # This saves the model to the trial directory


# This is a HyperOpt specific hyperparameter space configuration.
space = {
    "lr": hp.loguniform("lr", -10, -1),
    "momentum": hp.uniform("momentum", 0.1, 0.9),
}

# TODO: Create a HyperOptSearch object by passing in a HyperOpt specific search space.
# Also enforce that only 1 trials can run concurrently.
hyperopt_search = HyperOptSearch(
    space, max_concurrent=1, metric="mean_accuracy", mode="max"
)

# We Remove the dir so that we can visualize tensorboard correctly
# ! rm -rf ~/ray_results/search_algorithm


# ray.shutdown()  # Restart Ray defensively in case the ray connection is lost.

ray.init(address="auto")
# ray.init(num_gpus=6)
custom_scheduler = ASHAScheduler(
    metric="mean_accuracy",
    mode="max",
    grace_period=1,
)
analysis = tune.run(
    train_mnist,
    search_alg=hyperopt_search,
    scheduler=custom_scheduler,
    num_samples=10,
    resources_per_trial={"gpu": 1},
    verbose=1,
    name="use_gpu",  # This is used to specify the logging directory.
)
