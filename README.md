# Fluid: Resource-Aware Hyperparameter Tuning Engine

[![PyPI version](https://img.shields.io/pypi/v/fluidexec.svg)](https://pypi.org/project/fluidexec)
[![Build Status](https://github.com/SymbioticLab/fluid/actions/workflows/python-package/badge.svg?event=release)](https://github.com/SymbioticLab/fluid/actions)

`Fluid` is an alternative [Ray](https://ray.io) executor that intelligently manages trial executions on behalf of hyperparameter tuning algorithms, in order to increase the resource utilization, and improve end-to-end makespan.

This is the implementation of our MLSys'21 [paper](https://symbioticlab.org/publications/#/venue:MLSys) "Fluid: Resource-Aware Hyperparameter Tuning Engine".

## Get Started
First follow the [instruction](https://docs.ray.io/en/master/tune/index.html) in Ray Tune to setup the Ray cluster and a tuning environment as usual.

Then make sure [Nvidia MPS](https://docs.nvidia.com/deploy/mps/index.html#topic_6_1) is correctly setup on all worker nodes.

`Fluid` itself is a normal python package that can be installed by `pip install fluidexec`. Note that the pypi package name is `fluidexec` because the name `fluid` is already taken.

To use `Fluid` in Ray Tune, pass an instance of it as an additional keyword argument to `tune.run`:

```python
from fluid.executor import MyRayTrialExecutor
from fluid.scheduler import FluidBandScheduler
tune.run(
    MyTrainable,
    scheduler=FluidBandScheduler(...),
    trial_executor=FluidExecutor(),
    ...
)
```


## Reproduce Experiments
See the [README](workloads/README.md) in `workloads` for more information.


## Notes

Please consider to cite our paper if you find this useful in your research project.

```bibtex
@inproceedings{fluid:mlsys21,
    author    = {Peifeng Yu and Jiachen Liu and Mosharaf Chowdhury},
    booktitle = {MLSys},
    title     = {Fluid: Resource-Aware Hyperparameter Tuning Engine},
    year      = {2021},
}
```
