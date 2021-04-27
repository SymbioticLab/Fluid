# Experiments in Paper

## Clone the Repo
To run experiments, it is necessary to clone this repo, instad of installing `fluid` from PyPI.

`fluid` uses [`poetry`](https://python-poetry.org) to manage its dependencies, so `poetry` has to be
installed first.

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
git clone https://github.com/SymbioticLab/fluid
cd fluid
poetry install --extras "exp"
poetry shell
```

The last command spawns a new shell within the virtual environment just created with all dependencies
installed.

## Setup a Minimal Ray Cluster
[`cluster.yaml`](cluster.yaml) provides an example Ray cluster config file that you can use as the starting point to bring up a Ray cluster.

## Run Individual Experiments
Each `tune_*.py` file in this directory is an experiment that can run individualy.

To run them, first ensure the Ray cluster can be connected, then run `python -m workloads.tune_asha_cifar` from the root directory of the repo.

After it finishes, some statistical results will be saved in `trail_dataframe.csv`.
