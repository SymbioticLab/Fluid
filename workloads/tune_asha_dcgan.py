from pathlib import Path

from ray.tune.schedulers.async_hyperband import ASHAScheduler
from ray import tune

from fluid.algo_random import VariantGenerator
from fluid.trainer import TorchTrainer
import workloads.common as com
from workloads.common import dcgan as workload


DATA_PATH, RESULTS_PATH = com.detect_paths()
EXP_NAME = com.remove_prefix(Path(__file__).stem, 'tune_')


def setup_tune_scheduler():
    search_space = workload.create_search_space()

    scheduler = ASHAScheduler(
        # set a large max_t such that ASHA will always promot to next rung,
        # until something reaches target accuracy
        max_t=int(1000),
        reduction_factor=3,
        **workload.exp_metric()
    )
    return dict(
        search_alg=VariantGenerator(),
        scheduler=scheduler,
        config=search_space,
        resources_per_trial=com.detect_baseline_resource(),
    )


def main():
    _, sd = com.init_ray()
    workload.init_dcgan()

    MyTrainable = TorchTrainer.as_trainable(
        data_creator=workload.data_creator,
        model_creator=workload.model_creator,
        loss_creator=workload.loss_creator,
        optimizer_creator=workload.optimizer_creator,
        training_operator_cls=workload.GANOperator,
        config={
            'seed': sd,
            'extra_fluid_trial_resources': {},
            **workload.static_config(),
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
