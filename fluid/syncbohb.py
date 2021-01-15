#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:02:14 2020

@author: liujiachen
"""
from __future__ import annotations

import logging 
import numpy as np
from ray.tune.trial import Trial
from ray.tune.error import TuneError
from ray.tune.result import TIME_THIS_ITER_S ,TRAINING_ITERATION
from ray.tune.schedulers import  HyperBandForBOHB

from typing import NamedTuple, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Optional, TypedDict, Tuple, Union, Dict, Set
import os 
def sched_algo():
    return int(os.environ.get("NUM_WORKER", 4))


logger = logging.getLogger(__name__)

# Implementation notes:
#    Synchronous BOHB

class SyncBOHB(HyperBandForBOHB):
    def choose_trial_to_run(self, trial_runner):
        """Fair scheduling within iteration by completion percentage.

        List of trials not used since all trials are tracked as state
        of scheduler. If iteration is occupied (ie, no trials to run),
        then look into next iteration.
        """
        stop_find = False
        for hyperband in self._hyperbands:
            # band will have None entries if no resources
            # are to be allocated to that bracket.
            scrubbed = [b for b in hyperband if b is not None]
            for bracket in scrubbed:
                if any(t.status == Trial.PENDING or t.status == Trial.RUNNING for t in bracket.current_trials()) :
                    stop_find = True
                    if np.sum(list(map(lambda x: x.status == Trial.RUNNING, bracket.current_trials()))) < sched_algo() :
                        for trial in bracket.current_trials():
                            if (trial.status == Trial.PENDING
                                    and trial_runner.has_resources(trial.resources)):
                                return trial
                    break
            if stop_find:
                break

        # MAIN CHANGE HERE!
        # not a trial is running?
        if not any(t.status == Trial.RUNNING
                   for t in trial_runner.get_trials()):
            for hyperband in self._hyperbands:
                for bracket in hyperband:
                    if bracket and any(trial.status == Trial.PAUSED
                                       for trial in bracket.current_trials()):
                        # This will change the trial state and let the
                        # trial runner retry.
                        self._process_bracket(trial_runner, bracket)
        # MAIN CHANGE HERE!
        return None
