from pathlib import Path

from ray.tune.suggest.bohb import TuneBOHB
from ray import tune

from fluid.trainer import TorchTrainer
import workloads.common as com
from workloads.common import dcgan as workload


DATA_PATH, RESULTS_PATH = com.detect_paths()
EXP_NAME = com.remove_prefix(Path(__file__).stem, 'tune_')

from fluid.syncbohb import SyncBOHB


def setup_tune_scheduler(num_worker):

    # BOHB uses ConfigSpace for their hyperparameter search space
    config_space = workload.create_ch()

    experiment_metrics = workload.exp_metric()
    bohb_search = TuneBOHB(config_space, **experiment_metrics)

    bohb_hyperband = SyncBOHB(
        time_attr="training_iteration",
        max_t=81,
        reduction_factor=3,
        **experiment_metrics)

    return dict(
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        resources_per_trial=com.detect_baseline_resource(),
    )


def main():
    num_worker, sd = com.init_ray()
    workload.init_dcgan()

    MyTrainable_asha = TorchTrainer.as_trainable(
        data_creator=workload.data_creator,
        model_creator=workload.model_creator,
        loss_creator=workload.loss_creator,
        optimizer_creator=workload.optimizer_creator,
        training_operator_cls=workload.GANOperator,
        config={
            'seed': sd,
            **workload.static_config(),
            'extra_fluid_trial_resources': {}
        }
    )

    params = {
        **com.run_options(__file__),
        'stop': workload.create_stopper(),
        **setup_tune_scheduler(num_worker),
    }

    analysis = tune.run(
        MyTrainable_asha,
        **params
    )

    dfs = analysis.trial_dataframes
    for logdir, df in dfs.items():
        ld = Path(logdir)
        df.to_csv(ld / 'trail_dataframe.csv')


if __name__ == '__main__':
    main()
