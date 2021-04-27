#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 01:51:33 2020

@author: liujiachen
"""
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.trial import Trial
from ray.tune.trial_runner import TrialRunner
from ray.tune.web_server import TuneServer

class MyTrialRunner(TrialRunner) :

    def __init__(self,
                 search_alg=None,
                 scheduler=None,
                 launch_web_server=False,
                 local_checkpoint_dir=None,
                 remote_checkpoint_dir=None,
                 sync_to_cloud=None,
                 stopper=None,
                 resume=False,
                 server_port=TuneServer.DEFAULT_PORT,
                 fail_fast=False,
                 verbose=True,
                 checkpoint_period=10,
                 trial_executor=None):
        super(MyTrialRunner, self).__init__()

    def _execute_action(self, trial, decision):
        """Executes action based on decision.

        Args:
            trial (Trial): Trial to act on.
            decision (str): Scheduling decision to undertake.
        """
        if decision == TrialScheduler.CONTINUE:
            self.trial_executor.continue_training(trial)
        elif decision == TrialScheduler.PAUSE:
            self.trial_executor.pause_trial(trial)
        elif decision == TrialScheduler.STOP:
            self.trial_executor.export_trial_if_needed(trial)
            self.trial_executor.stop_trial(trial)
        elif decision == TrialScheduler.SCALEUP:
            assert trial.status == Trial.PENDING
            print("scale up")
        else:
            raise ValueError("Invalid decision: {}".format(decision))
