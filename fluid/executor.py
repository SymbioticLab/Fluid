# coding: utf-8
from ray.tune.error import TuneError
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.tune.trial import Trial

class MyRayTrialExecutor(RayTrialExecutor):

    def __init__(self):
        super(MyRayTrialExecutor, self).__init__()
    
    def on_no_available_trials(self, trial_runner):
        if self._queue_trials:
            return
        for trial in trial_runner.get_trials():
            if trial.status == Trial.PENDING:
                if not self.has_resources(trial.resources):
                    raise TuneError(
                        ("Insufficient cluster resources to launch trial: "
                         "trial requested {} but the cluster has only {}. "
                         "Pass `queue_trials=True` in "
                         "ray.tune.run() or on the command "
                         "line to queue trials until the cluster scales "
                         "up or resources become available. {}").format(
                             trial.resources.summary_string(),
                             self.resource_string(),
                             trial.get_trainable_cls().resource_help(
                                 trial.config)))
            elif trial.status == Trial.PAUSED:
                if not self.has_resources(trial.resources):
                    raise TuneError("There are paused trials, but no more pending "
                                    "trials with sufficient resources.")
