# Fluid: Resource-Aware Hyperparameter Tuning Engine

Fluid acts as an alternative [Ray](https://ray.io) executor.
To use it in Ray Tune, pass an instance as an additional keyword argument to `tune.run`:

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

