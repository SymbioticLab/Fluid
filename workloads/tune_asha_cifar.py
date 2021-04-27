import random
from pathlib import Path

import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.util.sgd.utils import BATCH_SIZE

import workloads.common as com
from fluid.algo_random import VariantGenerator
from fluid.trainer import TorchTrainer
from workloads.common import cifar as workload

DATA_PATH, RESULTS_PATH = com.detect_paths()
EXP_NAME = com.remove_prefix(Path(__file__).stem, "tune_")


def setup_tune_scheduler():
    search_space = workload.create_search_space()

    scheduler = ASHAScheduler(
        # set a large max_t such that ASHA will always promot to next rung,
        # until something reaches target accuracy
        max_t=int(1000),
        reduction_factor=3,
        **workload.exp_metric(),
    )
    return dict(
        search_alg=VariantGenerator(),
        scheduler=scheduler,
        config=search_space,
        resources_per_trial=com.detect_baseline_resource(),
    )


def main():
    eta, sd = com.init_ray()

    MyTrainable = TorchTrainer.as_trainable(
        data_creator=workload.data_creator,
        model_creator=workload.model_creator,
        loss_creator=workload.loss_creator,
        optimizer_creator=workload.optimizer_creator,
        config={BATCH_SIZE: 64, "seed": sd, "extra_fluid_trial_resources": {}},
    )

    params = {
        **com.run_options(__file__),
        "stop": workload.create_stopper(),
        **setup_tune_scheduler(),
    }

    analysis = tune.run(MyTrainable, **params)

    dfs = analysis.trial_dataframes
    for logdir, df in dfs.items():
        ld = Path(logdir)
        df.to_csv(ld / "trail_dataframe.csv")


if __name__ == "__main__":
    main()
