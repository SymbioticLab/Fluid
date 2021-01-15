#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 21:10:13 2020

@author: liujiachen
"""
from __future__ import annotations

import logging
import collections

import numpy as np
import ray
from ray.tune.trial import Trial
from ray.tune.result import TIME_THIS_ITER_S

from typing import NamedTuple, TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Optional, TypedDict, Dict

    class SchedState(TypedDict):
        bracket: Optional[Bracket]
        band_idx: int


logger = logging.getLogger(__name__)


class Bracket:
    """Logical object for tracking Hyperband bracket progress. Keeps track
    of proper parameters as designated by HyperBand.

    Also keeps track of progress to ensure good scheduling.
    """
    class RungParam(NamedTuple):
        idx: int
        # max number of trials
        n: int
        # max res allowed in this rung
        r: int
        # number of trials actually added
        added: int = 0

    class TrialInfo(NamedTuple):
        # the result
        result: dict
        # which rung this trial is on
        rung: int

    def __init__(self, res_name, max_trials, init_res, max_res, eta, s):
        # print("<<<< Init Bracket :",res_name, max_trials, init_res, max_res, eta, s )
        self._live_trials: Dict[Trial, Bracket.TrialInfo] = {}  # maps trial -> current result
        self._all_trials = []
        self._res_name = res_name  # attribute to

        self._eta = eta
        self._n = max_trials
        self._r = init_res
        self._max_res = max_res
        self._s = s
        # track progress for fair scheduling
        self._completed_progress = 0
        self._total_work = 0
        
        # pre-compute each rung's n and r
        n = max_trials
        r = init_res
        self._rungs = []
        for idx in range(s + 1):
            self._total_work += n * r
            rung = Bracket.RungParam(idx, n, r)
            self._rungs.append(rung)
            n = int(np.ceil(n / eta))
            r = int(min(r * eta, max_res ))
        
        self._halves = s

    '''
    def init_rung(max_trial, max_res):
        init_res = int(max_res / max_trial)
        num_iter = min(int(np.log(max_trial)/ np.log(self._eta) ),
                     int(np.log(max_res)/ np.log(self._eta) ) )
        for idx in range(num_iter):
    '''

    def add_trial(self, trial):
        """Add trial to bracket assuming bracket is not filled.

        At a later iteration, a newly added trial will be given equal
        opportunity to catch up."""
        self._live_trials[trial] = Bracket.TrialInfo(None, 0)
        self._add_trial_to_rung(trial, self._rungs[0])
        self._all_trials.append(trial)

    def _add_trial_to_rung(self, trial: Trial, rung: RungParam):
        assert not self.filled(rung.idx), "Cannot add trial to filled bracket!"
        
        self._rungs[rung.idx] = self._rungs[rung.idx] ._replace(added = rung.added+1)
        self._live_trials[trial]= self._live_trials[trial]._replace(rung = rung.idx)
        
    def filled(self, rung_idx: int = 0):
        """Checks if bracket is filled.

        Only let new trials be added at current level minimizing the need
        to backtrack and bookkeep previous medians."""
        if rung_idx >= len(self._rungs):
            return True

        return self._rungs[rung_idx].n == self._rungs[rung_idx].added

    def finished(self):
        """Bracket is finished if the last rung is done"""
        return self.rung_is_done(len(self._rungs) - 1)

    def rung_is_done(self, rung_idx):
        """Rung is done if
            - this rung is full and all done
            - next rung is full
            - next rung is done
        """
        if self.filled(rung_idx):
            # trials in rung may be fewer than rung.added, but it's fine.
            # the missing trials are promoted and can be considered terminated for this rung
            return all(not self.continue_trial(t) for t in self.trials_in_rung(rung_idx))
        return False

    def trials_in_rung(self, rung_idx: int) -> List[Trial]:
        return [t for t in self._live_trials if self._live_trials[t].rung == rung_idx]

    def current_trials(self):
        return list(self._live_trials)
    

    def cleanup_rungs(self) -> List[Trial]:
        """Return any trial that need to be terminated
        """
        clean_trials = [
            t
            for t, info in self._live_trials.items()
            if (
                self.rung_is_done(info.rung)
                and self.filled(info.rung + 1)
                and t.status != Trial.TERMINATED
            )
        ]
        return clean_trials

    def continue_trial(self, trial):
        result, rung_idx = self._live_trials[trial]
        rung = self._rungs[rung_idx]
        if self._get_result_time(result) < rung.r:
            return True
        else:
            return False

    def promotable(self, metric, metric_op) -> List[Trial]:
        promotables = []
        bad = []
        # going from last rung to rung 0
        for rung in self._rungs:
            trials_done = [
                t
                for t in self.trials_in_rung(rung.idx)
                if not self.continue_trial(t)
            ]
            
            # print("<<trial done: ", trials_done," / ", rung.n )
            if self.filled(rung.idx + 1) or len(trials_done) == 0 :
                continue
            
            nxt_trials_launch = [
                t
                for t in self.trials_in_rung(rung.idx + 1)
            ]
            
            if len(nxt_trials_launch) >= self._rungs[ rung.idx + 1 ].n:
                continue
            
            next_n = min( self._rungs[rung.idx + 1].n, self._n / (self._eta ** (rung.idx+1) ))
            cur_n =  min(rung.n  , self._n / (self._eta ** rung.idx))
            num_promotable = int( len(trials_done) - round(cur_n - next_n))
            
            if num_promotable <= 0:
                continue
            # sort in asc order, good ones are the last num_promotable trials
            sorted_trials = sorted(
                trials_done,
                key=lambda t: metric_op * self._live_trials[t].result[metric])
            promotables += sorted_trials[-num_promotable:]
            bad += sorted_trials[:-num_promotable]
        return promotables, bad 

    
    def promote(self, trial: Trial):
        next_idx = self._live_trials[trial].rung + 1
        new_rung = self._rungs[next_idx]
        self._add_trial_to_rung(trial, new_rung)

    def update_trial_stats(self, trial: Trial, result):
        """Update result for trial. Called after trial has finished
        an iteration - will decrement iteration count.

        TODO(rliaw): The other alternative is to keep the trials
        in and make sure they're not set as pending later."""

        assert trial in self._live_trials
        assert self._get_result_time(result) >= 0

        delta = self._get_result_time(result) - \
            self._get_result_time(self._live_trials[trial].result)
        assert delta >= 0, (result, self._live_trials[trial])
        self._completed_progress += delta
        
        self._live_trials[trial]= self._live_trials[trial]._replace(result=result)
        
    def cleanup_trial(self, trial):
        """Clean up statistics tracking for terminated trials (either by force
        or otherwise).

        This may cause bad trials to continue for a long time, in the case
        where all the good trials finish early and there are only bad trials
        left in a bracket with a large max-iteration."""
        assert trial in self._live_trials
        del self._live_trials[trial]

    def cleanup_full(self, trial_runner):
        """Cleans up bracket after bracket is completely finished.

        Lets the last trial continue to run until termination condition
        kicks in."""
        for trial in self.current_trials():
            if (trial.status == Trial.PAUSED):
                trial_runner.stop_trial(trial)

    def completion_percentage(self):
        """Returns a progress metric.

        This will not be always finish with 100 since dead trials
        are dropped."""
        if self.finished():
            return 1.0
        return self._completed_progress / self._total_work

    def _get_result_time(self, result):
        if result is None:
            return 0
        return result[self._res_name]

    def __repr__(self):
        status = ", ".join([
            "Max Sizes (n)={}".format([rung.n for rung in self._rungs]),
            "Milestones (r)={}".format([rung.r for rung in self._rungs]),
            "completed={:.1%}".format(self.completion_percentage())
        ])
        counts = collections.Counter([t.status for t in self._all_trials])
        trial_statuses = ", ".join(
            sorted("{}: {}".format(k, v) for k, v in counts.items()))
        return "Bracket({}): {{{}}} ".format(status, trial_statuses)

    def sort_by_runtime(self):
        self._live_trials = dict(sorted(self._live_trials.items(), key=lambda x:x[1].result[TIME_THIS_ITER_S], reverse=True))
        
        