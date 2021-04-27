import os
import urllib.request

import ConfigSpace as CS
import numpy as np
import torch
import torch.nn as nn
from ray import tune
from ray.util.sgd.torch.examples.dcgan import GANOperator as RayGANOperator
from ray.util.sgd.torch.examples.dcgan import model_creator, optimizer_creator
from ray.util.sgd.utils import BATCH_SIZE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import workloads.common as com

__all__ = [
    "data_creator",
    "model_creator",
    "loss_creator",
    "optimizer_creator",
    "GANOperator",
    "init_dcgan",
    "create_sample_space",
    "create_search_space",
    "create_ch",
    "create_stopper",
    "exp_metric",
    "static_config",
]


DATA_PATH, RESULTS_PATH = com.detect_paths()
MODEL_PATH = os.path.join(DATA_PATH, "models", "mnist_cnn.pt")


def static_config():
    return {
        "classification_model_path": MODEL_PATH,
    }


def exp_metric():
    # return dict(metric="loss_sum", mode="min")
    return dict(metric="inception", mode="max")


def create_stopper():
    return com.MetricStopper(5.3, **exp_metric())


def data_creator(config):
    dataset = datasets.MNIST(
        DATA_PATH,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=config.get(BATCH_SIZE, 32))
    return dataloader, []


def loss_creator(config):
    return nn.BCELoss()


class GANOperator(RayGANOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lastD = None
        self._lastG = None

    def _should_update_D(self, errD):
        if self._lastG is None:
            return True
        self._lastD = errD
        return errD >= 0.5 * self._lastG

    def _should_update_G(self, errG):
        if self._lastD is None:
            return True
        self._lastG = errG
        return errG >= 0.5 * self._lastD

    def train_batch(self, batch, batch_info):
        """Trains on one batch of data from the data creator."""
        real_label = 1
        fake_label = 0
        discriminator, generator = self.models
        optimD, optimG = self.optimizers

        # Compute a discriminator update for real images
        discriminator.zero_grad()
        # self.device is set automatically
        real_cpu = batch[0].to(self.device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=self.device)
        output = discriminator(real_cpu).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()

        # Compute a discriminator update for fake images
        noise = torch.randn(
            batch_size,
            self.config.get("latent_vector_size", 100),
            1,
            1,
            device=self.device,
        )
        fake = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake

        # Update the discriminator
        # only update D if it can not get half the fake label correct
        # i.e. less than 50% of label is zero
        if self._should_update_D(errD):
            optimD.step()

        # Update the generator
        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake).view(-1)
        errG = self.criterion(output, label)
        errG.backward()

        # only update G if more than 50% of label is zero
        if self._should_update_G(errG):
            optimG.step()

        is_score, is_std = self.inception_score(fake)

        return {
            "loss_g": errG.item(),
            "loss_d": errD.item(),
            "inception": is_score,
            "num_samples": batch_size,
        }

    def validate(self, batch, info):
        return {}


def init_dcgan():
    # Download a pre-trained MNIST model for inception score calculation.
    # This is a tiny model (<100kb).
    if not os.path.exists(MODEL_PATH):
        print("downloading model")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/ray-project/ray/raw/master/python/ray/tune/"
            "examples/pbt_dcgan_mnist/mnist_cnn.pt",
            MODEL_PATH,
        )


def create_sample_space():
    mutations = {
        "netG_lr": lambda *_: np.random.uniform(0.0001, 0.0005),
        "netD_lr": lambda *_: np.random.uniform(0.0001, 0.0005),
    }

    cap_explore = com.create_cap_explore_fn(
        mutations,
        [
            ("netG_lr", 0.0001, 0.0005),
            ("netD_lr", 0.0001, 0.0005),
        ],
    )

    return mutations, cap_explore


def create_search_space():
    return {key: tune.sample_from(val) for key, val in create_sample_space()[0].items()}


def create_ch():
    # BOHB uses ConfigSpace for their hyperparameter search space
    config_space = CS.ConfigurationSpace()

    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(name="netG_lr", lower=0.0001, upper=0.0005)
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(name="netD_lr", lower=0.0001, upper=0.0005)
    )

    return config_space


def create_skopt_space():
    return [(0.0001, 0.0005), (0.0001, 0.0005),], [
        "netG_lr",
        "netD_lr",
    ]
