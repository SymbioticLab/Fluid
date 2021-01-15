from pathlib import Path

from ray.util.sgd.utils import BATCH_SIZE
from ray import tune

from fluid.executor import MyRayTrialExecutor
from fluid.scheduler import FluidBandScheduler
from fluid.trainer import TorchTrainer
import workloads.common as com
from workloads.common import mnist as workload


DATA_PATH, RESULTS_PATH = com.detect_paths()
EXP_NAME = com.remove_prefix(Path(__file__).stem, 'tune_')


def setup_tune_scheduler():
    from ray.tune.suggest.skopt import SkOptSearch
    from ray.tune.suggest.suggestion import ConcurrencyLimiter
    from skopt import Optimizer

    exp_metrics = workload.exp_metric()

    search_space, dim_names = workload.create_skopt_space()
    algo = ConcurrencyLimiter(SkOptSearch(
        Optimizer(search_space), dim_names,
        **exp_metrics,
    ), 3)

    scheduler = FluidBandScheduler(
        max_res=3,
        reduction_factor=3,
        **exp_metrics,
    )
    return dict(
        search_alg=algo,
        scheduler=scheduler,
        trial_executor=MyRayTrialExecutor(),
        resources_per_trial=com.detect_baseline_resource(),
    )


def main():
    _, sd = com.init_ray()

    MyTrainable = TorchTrainer.as_trainable(
        data_creator=workload.data_creator,
        model_creator=workload.model_creator,
        loss_creator=workload.loss_creator,
        optimizer_creator=workload.optimizer_creator,
        config={
            'seed': sd,
            BATCH_SIZE: 64,
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
