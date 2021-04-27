import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

import ray
from ray import tune
from ray.tune import track, Trainable
from ray.tune.schedulers import ASHAScheduler
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test



from fluid.scheduler import FluidBandScheduler
from fluid.executor import MyRayTrialExecutor



def train_mnist(config):
    model = ConvNet()
    train_loader, test_loader = get_data_loaders()
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for i in range(10):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        track.log(mean_accuracy=acc)
        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model, "./model.pth")

def train_mnist_bohb(config, model=None):
    if model is None:
        model = ConvNet()
    train_loader, test_loader = get_data_loaders()
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for i in range(10):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        # track.log(mean_accuracy=acc)
        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model, "./model.pth")
    return acc


class MyTrainableClass(Trainable):
    """Example agent whose learning curve is a random sigmoid.
    The dummy hyperparameters "width" and "height" determine the slope and
    maximum reward value reached.
    """

    def _setup(self, config):
        self.timestep = 0
        self.model = ConvNet()

    def _train(self):
        self.timestep += 1
        acc = train_mnist_bohb(self.config, self.model)

        return {"mean_accuracy": acc}

    def _save(self, checkpoint_dir):

        import os
        path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), path)
        return path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


def init_ray():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="localhost:6379")
    args = parser.parse_args()
    # ray.init(address=args.ray_address)
    ray.init()

def main():
    try:
        init_ray()

        search_space = {
            "lr": tune.sample_from(lambda spec: 10**(-10 * np.random.rand())),
            "momentum": tune.uniform(0.1, 0.9)
        }

        analysis = tune.run(
            MyTrainableClass,
            trial_executor = MyRayTrialExecutor(),
            num_samples=9,
            scheduler = FluidBandScheduler(),
            #scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
            config=search_space
        )

        dfs = analysis.trial_dataframes
        for logdir, df in dfs.items():
            ld = Path(logdir)
            df.to_csv(ld / 'trail_dataframe.csv')
    finally:
        ray.shutdown()

if __name__ == '__main__':
    main()
