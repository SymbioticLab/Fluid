#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from ray import tune
from ray.util.sgd.utils import BATCH_SIZE

import workloads.common as com
from fluid.algo_random import VariantGenerator
from fluid.ashaparallel import AsyncHyperBandSchedulerWithParalelism
from fluid.trainer import TorchTrainer
from workloads.common import cifar as workload

DATA_PATH, RESULTS_PATH = com.detect_paths()
EXP_NAME = com.remove_prefix(Path(__file__).stem, "tune_")

import os


def sched_algo():
    return int(os.environ.get("NUM_WORKER", 4))


def setup_tune_scheduler(num_worker):
    search_space = workload.create_search_space()
    # experiment_metrics = workload.exp_metric()
    asha_parallel = AsyncHyperBandSchedulerWithParalelism(
        # set a large max_t such that ASHA will always promot to next rung,
        # until something reaches target accuracy
        max_t=int(1000),
        reduction_factor=3,
        **workload.exp_metric()
    )

    return dict(
        scheduler=asha_parallel,
        search_alg=VariantGenerator(max_concurrent=sched_algo()),
        config=search_space,
        resources_per_trial=com.detect_baseline_resource(),
    )


def main():
    num_worker, sd = com.init_ray()

    MyTrainable_SyncBOHB = TorchTrainer.as_trainable(
        data_creator=workload.data_creator,
        model_creator=workload.model_creator,
        loss_creator=workload.loss_creator,
        optimizer_creator=workload.optimizer_creator,
        config={"seed": sd, BATCH_SIZE: 64, "extra_fluid_trial_resources": {}},
    )

    params = {
        **com.run_options(__file__),
        "stop": workload.create_stopper(),
        **setup_tune_scheduler(num_worker),
    }

    analysis = tune.run(MyTrainable_SyncBOHB, **params)

    dfs = analysis.trial_dataframes
    for logdir, df in dfs.items():
        ld = Path(logdir)
        df.to_csv(ld / "trail_dataframe.csv")


if __name__ == "__main__":
    main()
