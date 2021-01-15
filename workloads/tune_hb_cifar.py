from pathlib import Path

from ray.tune.schedulers.hb_bohb import HyperBandScheduler
from ray.util.sgd.utils import BATCH_SIZE
from ray import tune

from fluid.algo_random import VariantGenerator
from fluid.trainer import TorchTrainer
import workloads.common as com
from workloads.common import cifar as workload

import numpy as np
import random
import torch

DATA_PATH, RESULTS_PATH = com.detect_paths()
EXP_NAME = com.remove_prefix(Path(__file__).stem, 'tune_')


def setup_tune_scheduler(eta):
    search_space = workload.create_search_space()

    experiment_metrics = workload.exp_metric()
    scheduler = HyperBandScheduler(
        time_attr="training_iteration",
        max_t=81,
        reduction_factor=eta,
        **experiment_metrics)

    return dict(
        search_alg=VariantGenerator(),
        scheduler=scheduler,
        config=search_space,
        resources_per_trial=com.detect_baseline_resource(),
    )


def main():
    eta, sd = com.init_ray()
    eta = 3 if eta == 1 else eta
    
    MyTrainable = TorchTrainer.as_trainable(
        data_creator=workload.data_creator,
        model_creator=workload.model_creator,
        loss_creator=workload.loss_creator,
        optimizer_creator=workload.optimizer_creator,
        config={
            'seed':sd,
            BATCH_SIZE: 64,
            'extra_fluid_trial_resources': {}
        }
    )

    params = {
        **com.run_options(__file__),
        'stop': workload.create_stopper(),
        **setup_tune_scheduler(eta),
    }

    analysis = tune.run(
        MyTrainable,
        **params
    )

    dfs = analysis.trial_dataframes
    for logdir, df in dfs.items():
        ld = Path(logdir)
        df.to_csv(ld / 'trail_dataframe.csv')


if __name__ == '__main__':
    main()
