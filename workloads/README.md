# Experiments in Paper

## Setup a Minimal Ray Cluster
[`cluster.yaml`](cluster.yaml) provides an example Ray cluster config file that you can use as the starting point to bring up a Ray cluster.

## Run Individual Experiments
Each `tune_*.py` file in this directory is an experiment that can run individualy.

To run them, first ensure the Ray cluster can be connected, then run `python -m workloads.tune_asha_cifar` from the root directory of the repo.

After it finishes, some statistical results will be saved in `trail_dataframe.csv`.
