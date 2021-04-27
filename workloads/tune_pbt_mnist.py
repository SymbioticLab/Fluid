from pathlib import Path

from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.util.sgd.utils import BATCH_SIZE

import workloads.common as com
from fluid.trainer import TorchTrainer
from workloads.common import mnist as workload

DATA_PATH, RESULTS_PATH = com.detect_paths()
EXP_NAME = com.remove_prefix(Path(__file__).stem, "tune_")


def setup_tune_scheduler():
    ss, custom_explore = workload.create_sample_space()
    search_space = workload.create_search_space()

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations=ss,
        custom_explore_fn=custom_explore,
        **workload.exp_metric()
    )

    return dict(
        scheduler=scheduler,
        config=search_space,
        # num_samples in PBT only sets population
        num_samples=10,
        resources_per_trial=com.detect_baseline_resource(),
    )


def main():
    _, sd = com.init_ray()

    MyTrainable = TorchTrainer.as_trainable(
        data_creator=workload.data_creator,
        model_creator=workload.model_creator,
        loss_creator=workload.loss_creator,
        optimizer_creator=workload.optimizer_creator,
        config={"seed": sd, BATCH_SIZE: 64, "extra_fluid_trial_resources": {}},
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
