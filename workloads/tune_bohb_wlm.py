from pathlib import Path

from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray import tune

from fluid.trainer import TorchTrainer
import workloads.common as com
from workloads.common import wlm as workload


import numpy as np
import random
import torch

DATA_PATH, RESULTS_PATH = com.detect_paths()
EXP_NAME = com.remove_prefix(Path(__file__).stem, 'tune_')


def setup_tune_scheduler():
    config_space = workload.create_ch()

    experiment_metrics = workload.exp_metric()
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=243,
        reduction_factor=3,
        **experiment_metrics
    )
    bohb_search = TuneBOHB(
        config_space, **experiment_metrics)

    return dict(
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
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
        **params
    )

    dfs = analysis.trial_dataframes
    for logdir, df in dfs.items():
        ld = Path(logdir)
        df.to_csv(ld / 'trail_dataframe.csv')


if __name__ == '__main__':
    main()
