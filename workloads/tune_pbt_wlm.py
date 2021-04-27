from pathlib import Path

from ray.tune.schedulers.pbt import PopulationBasedTraining
from ray import tune

from fluid.trainer import TorchTrainer
import workloads.common as com
from workloads.common import wlm as workload


DATA_PATH, RESULTS_PATH = com.detect_paths()
EXP_NAME = com.remove_prefix(Path(__file__).stem, 'tune_')

import numpy as np
import random
import torch


def setup_tune_scheduler():
    ss, custom_explore = workload.create_sample_space()
    tune_space = workload.create_search_space()

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=200,
        hyperparam_mutations=ss,
        custom_explore_fn=custom_explore,
        **workload.exp_metric(),
    )

    return dict(
        scheduler=scheduler,
        config=tune_space,
        # num_samples in PBT only sets population
        num_samples=8,
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
        training_operator_cls=workload.WLMOperator,
        config={
            'seed': sd,
            'extra_fluid_trial_resources': {}
        }
    )

    params = {
        **com.run_options(__file__),
        'stop': workload.create_stopper(),
        **setup_tune_scheduler(),
    }

    analysis = tune.run(
        MyTrainable,
        checkpoint_freq=40,
        **params
    )

    dfs = analysis.trial_dataframes
    for logdir, df in dfs.items():
        ld = Path(logdir)
        df.to_csv(ld / 'trail_dataframe.csv')


if __name__ == '__main__':
    main()
