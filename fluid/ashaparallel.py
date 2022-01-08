#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 04:16:34 2020

@author: liujiachen
"""
import logging
import os

import numpy as np
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.trial import Trial

logger = logging.getLogger(__name__)


def sched_algo():
    return int(os.environ.get("NUM_WORKER", 8))


class AsyncHyperBandSchedulerWithParalelism(AsyncHyperBandScheduler):
    def choose_trial_to_run(self, trial_runner):
        for trial in trial_runner.get_trials():
            if trial.status == Trial.PENDING and trial_runner.has_resources(
                trial.resources
            ):
                if (
                    np.sum(
                        list(
                            map(
                                lambda x: x.status == Trial.RUNNING,
                                trial_runner.get_trials(),
                            )
                        )
                    )
                    < sched_algo()
                ):
                    return trial
        for trial in trial_runner.get_trials():
            if trial.status == Trial.PAUSED and trial_runner.has_resources(
                trial.resources
            ):
                if (
                    np.sum(
                        list(
                            map(
                                lambda x: x.status == Trial.RUNNING,
                                trial_runner.get_trials(),
                            )
                        )
                    )
                    < sched_algo()
                ):
                    return trial
        return None
