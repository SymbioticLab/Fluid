from __future__ import annotations

import logging
import time
import numpy as np
import ray
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.trial import Trial
from ray.tune.error import TuneError
from ray.tune.result import TIME_THIS_ITER_S, TRAINING_ITERATION
from .my_bracket import Bracket
from .ray_custom_gpu_res import create_custom_gpu_res
from collections import defaultdict

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Optional, TypedDict, Tuple, Union, Dict

    class SchedState(TypedDict):
        bracket: Optional[Bracket]
        band_idx: int


logger = logging.getLogger(__name__)


# Implementation notes:
#    This implementation contains 3 logical levels.
#    Each HyperBand iteration is a "band". There can be multiple
#    bands running at once, and there can be 1 band that is incomplete.
#
#    In each band, there are at most `s` + 1 brackets.
#    `s` is a value determined by given parameters, and assigned on
#    a cyclic basis.
#
#    In each bracket, there are at most `n(s)` trials, indicating that
#    `n` is a function of `s`. These trials go through a series of
#    halving procedures, dropping lowest performers. Multiple
#    brackets are running at once.
#
#    Trials added will be inserted into the most recent bracket
#    and band and will spill over to new brackets/bands accordingly.
#
#    This maintains the bracket size and max trial count per band
#    to 5 and 117 respectively, which correspond to that of
#    `max_attr=81, eta=3` from the blog post. Trials will fill up
#    from smallest bracket to largest, with largest
#    having the most rounds of successive halving.
class FluidBandScheduler(FIFOScheduler):
    """Implements the FluidBand early stopping algorithm.

    HyperBandScheduler early stops trials using the HyperBand optimization
    algorithm. It divides trials into brackets of varying sizes, and
    periodically early stops low-performing trials within each bracket.

    To use this implementation of HyperBand with Tune, all you need
    to do is specify the max length of time a trial can run `max_t`, the time
    units `time_attr`, the name of the reported objective value `metric`,
    and if `metric` is to be maximized or minimized (`mode`).
    We automatically determine reasonable values for the other
    HyperBand parameters based on the given values.

    For example, to limit trials to 10 minutes and early stop based on the
    `episode_mean_reward` attr, construct:

    ``HyperBand('time_total_s', 'episode_reward_mean', max_t=600)``

    Note that Tune's stopping criteria will be applied in conjunction with
    HyperBand's early stopping mechanisms.

    See also: https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

    Args:
        time_attr (str): The training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        metric (str): The training result objective value attribute. Stopping
            procedures will use this attribute.
        mode (str): One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        max_t (int): max time units per trial. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
            The scheduler will terminate trials after this time has passed.
            Note that this is different from the semantics of `max_t` as
            mentioned in the original HyperBand paper.
        reduction_factor (float): Same as `eta`. Determines how sharp
            the difference is between bracket space-time allocation ratios.
    """
    def __init__(self,
                 res_name="training_iteration",
                 max_res=9,
                 metric="mean_accuracy",
                 mode="max",
                 reduction_factor=3):
        assert max_res > 0, "Max (time_attr) not valid!"
        assert mode in ["min", "max"], "`mode` must be 'min' or 'max'!"

        # super().__init__(self)
        FIFOScheduler.__init__(self)
        self._eta = reduction_factor
        # number of interations
        self._s_max_1 = int(
            np.round(np.log(max_res) / np.log(reduction_factor))) + 1
        self._max_res = max_res
        # bracket max trials
        self._get_n0 = lambda s: int(
            np.ceil(self._s_max_1 / (s + 1) * self._eta**s))
        # bracket initial iterations
        self._get_r0 = lambda s: int((max_res * self._eta**(-s)))
        # list of hyperband iterations
        self._iterations: List[List[Optional[Bracket]]] = [[]]
        # Stores Trial -> Bracket, Band Iteration
        self._trial_info: Dict[Trial, Tuple[Bracket, int]] = {}

        self._num_stopped = 0
        self._metric = metric
        if mode == "max":
            self._metric_op = 1.
        elif mode == "min":
            self._metric_op = -1.
        self._res_name = res_name

        # scheduler estimator
        self.base_rung_runtime_sum = [
            [0 for x in range(self._s_max_1)]
        ]
        # map trial to init res
        self._res_estimator = {}
        self.trial_runtime = [[]]
        # scheduler state machine
        self.stage = 0
        # get our own GPU resources, should be created first
        create_custom_gpu_res()
        time.sleep(2)
        self._gpu_resources: Dict[str, float] = {
            k: v
            for k, v in ray.cluster_resources().items()
            if k.startswith('fluid_GPU')
        }
        self.num_gpu_resources = len(self._gpu_resources)
        self.gpu_overhead = [0 for x in range(self.num_gpu_resources)]
        self.customized_gpu_wl: List[Dict] = [
            dict() for x in range(self.num_gpu_resources)
        ]
        self.trial_overhead = defaultdict(int)
        self.update_res = {}  # recored updated res for scaling up
        self.cur_iter_idx = 0
        self.cur_braket_idx = 0
        self.start_iter = [[]]
        print("[Grab {num} resources]:{gpu} ".format(
            num=self.num_gpu_resources, gpu=self._gpu_resources))

        self._sm: Union[None, Stage0, Stage1, Stage2] = None  # Stage0(self)
        logger.info("[Grab resources] {res}".format(res=self._gpu_resources))

    def _update_cur_resource_usage(self, trial, result):
        pre_result = self._res_estimator[trial]
        orig_time = pre_result[TIME_THIS_ITER_S]
        cur_time = result[TIME_THIS_ITER_S]
        self.trial_overhead[trial] = max((cur_time - orig_time) / orig_time, 0)
        for res, num in trial.resources.custom_resources.items():
            assert res in self._gpu_resources
            gpu_idx = list(self._gpu_resources).index(res)
            self.customized_gpu_wl[gpu_idx][trial] = num
            self.gpu_overhead[gpu_idx] = 0
            for t in self.customized_gpu_wl[gpu_idx]:
                self.gpu_overhead[gpu_idx] += self.trial_overhead[t]

        print(">>GPU overhead:", self.gpu_overhead)
        print(">>GPU usage:", self.customized_gpu_wl)

    def _clear_cur_resource_usage(self, trial):
        self.trial_overhead[trial] = 0
        for res, num in trial.resources.custom_resources.items():
            assert res in self._gpu_resources
            gpu_idx = list(self._gpu_resources).index(res)
            if trial in self.customized_gpu_wl[gpu_idx]:
                del self.customized_gpu_wl[gpu_idx][trial]

                self.gpu_overhead[gpu_idx] = 0
                for t in self.customized_gpu_wl[gpu_idx]:
                    self.gpu_overhead[gpu_idx] += self.trial_overhead[t]

    def _find_first_avail_bracket(self) -> Tuple[Bracket, int]:
        def add_bracket(iter_idx, iteration) -> Optional[Bracket]:
            s = self._s_max_1 - 1
            if self._get_r0(s) == 0:
                return None
            b = Bracket(self._res_name, self._get_n0(s), self._get_r0(s),
                        self._max_res, self._eta, s)

            iteration.append(b)
            self.start_iter[iter_idx].append(0)
            self.trial_runtime[iter_idx].append([])
            return b

        for iter_idx, iteration in enumerate(self._iterations):
            for b in iteration:
                if b is None:
                    continue
                if b.filled():
                    continue
                return b, iter_idx
            else:
                # no existing bracket available
                # try add new brackets until filled
                while len(iteration) < self._s_max_1:
                    # create new one,
                    b = add_bracket(iter_idx, iteration)
                    if b is not None:
                        return b, iter_idx
            # now this iteration must have been filled,
            # o.w. the func should already returned
            assert len(iteration) == self._s_max_1

        # no existing iter available, create new one
        iter_idx = len(self._iterations)
        self._iterations.append([])
        self.base_rung_runtime_sum.append([0 for x in range(self._s_max_1)])
        self.start_iter.append([])
        self.trial_runtime.append([])
        b = add_bracket(iter_idx, self._iterations[iter_idx])
        assert b is not None
        return b, iter_idx

    def on_trial_add(self, trial_runner, trial):
        """Adds new trial.

        Find first bracket that is not filled to add the trial in all iterations,
        create new bracket/iteration if necessary.
        """

        b, iter_idx = self._find_first_avail_bracket()
        b.add_trial(trial)
        self._trial_info[trial] = b, iter_idx

    def choose_trial_to_run(self, trial_runner):
        """
        While being misleading, this method doesn't really schedules trials.
        Here it will simply return any trial that is pending.

        The actual scheduling happens in on_trial_result, which considers trials
        and resources.
        """
        stop_find = False

        for idx, iteration in enumerate(self._iterations):

            # band will have None entries if no resources
            # are to be allocated to that bracket.
            scrubbed = [b for b in iteration if b is not None]
            for indx_b, bracket in enumerate(scrubbed):
                if any(t.status == Trial.PENDING or t.status == Trial.RUNNING
                       for t in bracket.current_trials()):
                    stop_find = True
                    for trial in bracket.current_trials():
                        if (trial.status == Trial.PENDING and
                                trial_runner.has_resources(trial.resources)):
                            if trial in self.update_res:
                                self.update_resource(self.update_res[trial],
                                                     trial)
                                del self.update_res[trial]
                            if self.start_iter[idx][indx_b] == 0:
                                self.start_iter[idx][indx_b] = 1
                                self.stage = 0
                                self._sm = Stage0(self)
                            print("choose_trial_to_run: ", trial, " on ",
                                  trial.resources)
                            return trial
                    break
            if stop_find:
                break

        _, _, cur_bracket = self.find_cur_braket()

        if self.stage == 0 and \
                all([t.status == Trial.PAUSED for t in cur_bracket.current_trials()]):
            self.stage += 1
            self._sm = Stage1(self, trial_runner)

        return None

    def on_trial_result(self, trial_runner, trial: Trial, result) -> str:
        """
        This is called whenever a trial makes progress.
        We can make decision about whether to continue/pause/stop this trial by returning
        TrialScheduler.CONTINUE / TrialScheduler.PAUSE / TrialScheduler.STOP

        While we may also want to continue/pause/stop other trials,
        other running trials has to be paused/stopped in their corresponding
        on_trial_result.
        to unpause trials, call self._unpause_trial(t)
        """
        bracket, _ = self._trial_info[trial]
        bracket.update_trial_stats(trial, result)
        if self.stage > 0:
            self._update_cur_resource_usage(trial, result)

        # the iteration is a state machine
        # stage0: profiling stage, run each trial once and then pause
        # stage1: offline packing
        # stage2: online scheduling
        return self._sm.on_trial_result(trial_runner, trial, result)

    def on_trial_remove(self, trial_runner, trial):
        """Notification when trial terminates.

        Trial info is removed from bracket. Triggers halving if bracket is
        not finished."""
        bracket, _ = self._trial_info[trial]
        bracket.cleanup_trial(trial)
        if self.stage > 1:
            self._clear_cur_resource_usage(trial)
        return self._sm.on_trial_remove(trial_runner, trial)

    def on_trial_complete(self, trial_runner, trial, result):
        """Cleans up trial info from bracket if trial completed early."""
        self.on_trial_remove(trial_runner, trial)

    def on_trial_error(self, trial_runner, trial):
        """Cleans up trial info from bracket if trial errored early."""
        self.on_trial_remove(trial_runner, trial)

    def debug_string(self):
        """This provides a progress notification for the algorithm.

        For each bracket, the algorithm will output a string as follows:

            Bracket(Max Size (n)=5, Milestone (r)=33, completed=14.6%):
            {PENDING: 2, RUNNING: 3, TERMINATED: 2}

        "Max Size" indicates the max number of pending/running experiments
        set according to the Hyperband algorithm.

        "Milestone" indicates the iterations a trial will run for before
        the next halving will occur.

        "Completed" indicates an approximate progress metric. Some brackets,
        like ones that are unfilled, will not reach 100%.
        """
        out = "Using FluidBand: "
        out += "num_stopped={} total_brackets={}".format(
            self._num_stopped, sum(len(band) for band in self._iterations))
        for i, band in enumerate(self._iterations):
            out += "\nRound #{}:".format(i)
            for bracket in band:
                out += "\n  {}".format(bracket)
                # out += "\n {}".format(bracket._live_trials)
        return out

    def state(self):
        return {
            "num_brackets": sum(len(band) for band in self._iterations),
            "num_stopped": self._num_stopped
        }

    def find_cur_braket(self):
        for idx, iteration in enumerate(self._iterations):
            scrubbed = [b for b in iteration if b is not None]
            for idx_b, bracket in enumerate(scrubbed):
                if bracket.completion_percentage() < 1:
                    return idx, idx_b, bracket

    def _unpause_trial(self, trial_runner, trial):
        # set_status(Trial.PAUSE --> Trial.PENDING)
        trial_runner.trial_executor.unpause_trial(trial)

    def update_resource(self, gpu_dict, trial):
        assert trial.status != Trial.RUNNING
        # split the gpu_dict into one unit of resources, and remaining
        first, *others = gpu_dict.items()
        first = dict([first])
        others = dict(others)
        # the first one unit of resources is applied to the trial directly,
        # while the remaining are passed in as extra resources, which affects
        # scheduling, but are not allocated on the trail's ray actor.
        # In the trail's ray actor (see trainer.py _start_workers), more ray actors
        # are started, which will use the extra resources.
        if ray.cluster_resources().get('GPU', 0) > 0:
            trial.update_resources(
                cpu=1,
                gpu=sum(first.values()),
                custom_resources=first,
                extra_gpu=sum(others.values()),
                extra_custom_resources=others,
            )
        else:
            trial.update_resources(
                cpu=sum(first.values()),
                gpu=0,
                custom_resources=first,
                extra_cpu=sum(others.values()),
                extra_custom_resources=others,
            )
        # this config key is must have for TorchTrainer to work correctly
        # this is used to pass the extr resource dict to the trainer.
        # there is no other way to do this at the moment.
        trial.config['extra_fluid_trial_resources'] = others
        # this is for logging purposes.
        trial.config['all_fluid_trial_resources'] = gpu_dict


class Stage0:
    '''First run every trial for one step to get performance data
    '''
    def __init__(self, sched: FluidBandScheduler):
        self.p = sched
        print("[Stage0 init: Profiling jobs]")
        self.p.gpu_overhead = [0 for x in range(self.p.num_gpu_resources)]
        self.p.customized_gpu_wl: List[Dict] = [
            dict() for x in range(self.p.num_gpu_resources)
        ]
        self.p.trial_overhead = defaultdict(int)
        self.p.update_res = {}  # recored updated res for scaling up
        self.cur_iter, self.cur_b, self.b = self.p.find_cur_braket()

    def on_trial_result(self, trial_runner, trial, result) -> str:
        self.p._res_estimator[trial] = result
        assert result.get(TIME_THIS_ITER_S) is not None
        self.p.trial_runtime[self.cur_iter][self.cur_b].append(
            result[TIME_THIS_ITER_S])
        self.p.base_rung_runtime_sum[self.cur_iter][
            self.cur_b] += result[TIME_THIS_ITER_S]

        b, _ = self.p._trial_info[trial]

        # num_failed = len([t for t in b.current_trials() if t.error_file])
        logger.info("Profiling trial {idx}".format(idx=trial.trial_id))
        # pause to wait, jobs will be resumed in stage 1
        # trial.status == Trial.PAUSE
        print("Profiling trial ", trial.trial_id)
        return TrialScheduler.PAUSE

    def on_trial_remove(self, trial_runner, trial):
        pass


class Stage1:
    def __init__(self, sched: FluidBandScheduler, trial_runner):
        print("[Stage1 init: Allocating resources]")
        self.p = sched
        # only supports run one bracket in one iteration for now
        self.b = self.p._iterations[-1][-1]
        assert isinstance(self.b, Bracket)
        self.b._n = len(self.b.current_trials())
        self.b._s = min(int(np.log(self.b._n) / np.log(self.b._eta)),
                        self.b._s)
        self.cur_iter = len(self.p._iterations) - 1
        self._num_worker = self.p.num_gpu_resources
        # pending actions
        self._pending_actions: Dict[Trial, str] = {}
        self.cur_iter, self.cur_b, self.b = self.p.find_cur_braket()

        # sort current itereation rung 0 trials based on runtime
        self.b.sort_by_runtime()
        self.p.trial_runtime[self.cur_iter][self.cur_b].sort(reverse=True)
        print("[Runtime] from stage 0: ", self.p.trial_runtime[self.cur_iter])
        # assign resource to each trail
        self.max_trial = self.b._n
        # MPS packing
        self.MAX_WORKER = 4  # HARDCODE optimal worker
        # evenly dispatch & balance
        # only support for place two model on one gpu
        if self.max_trial > self._num_worker:
            self._mps_packing(trial_runner)
        # Distributed Bin Packing
        elif self.max_trial == self._num_worker:
            self._dist_bin_packing(trial_runner)
        else:
            self._even_bin_packing(trial_runner)

        if all([t.status == Trial.PAUSED for t in self.b.current_trials()]):
            self.p.stage = 2
            self.p._sm = Stage2(self.p, trial_runner)

    def _launch_trial(self, trial, trial_runner, gpu_dict):

        self.p.update_resource(gpu_dict, trial)
        if self.b.continue_trial(trial):
            self.p._unpause_trial(trial_runner, trial)

    def _mps_packing(self, trial_runner):
        print("[MPS]")
        resource_threshold = 0.5 * self.p.base_rung_runtime_sum[self.cur_iter][
            self.cur_b]
        sum_for_th = 0
        for idx, trial in enumerate(self.b.current_trials()):
            res = 0.4 if sum_for_th > resource_threshold else 0.6
            odd_even = int(idx / self._num_worker) % 2
            gpu_idx = (idx % self._num_worker)*((-1) ** odd_even) + odd_even * (self._num_worker - 1)
            gpu_dict = {list(self.p._gpu_resources.keys())[int(gpu_idx)]: res}
            print("[MPS] Place trial {idx} on {gpu} : {res}".format(
                idx=trial.trial_id,
                res=res,
                gpu=list(self.p._gpu_resources.keys())[int(gpu_idx)]))
            sum_for_th += self.p.trial_runtime[self.cur_iter][self.cur_b][idx]
            self._launch_trial(trial, trial_runner, gpu_dict)

    def _dist_bin_packing(self, trial_runner):
        print("[Bin Packing]")
        topk_num = int(self._num_worker / 2)
        start_gpu = 0
        for idx, trial in enumerate(self.b.current_trials()):
            res = 1
            gpu_dict = {}
            if idx < topk_num:
                small_job_sum_time = 0
                for i in range(self._num_worker - topk_num):
                    small_job_sum_time += self.p.trial_runtime[self.cur_iter][
                        self.cur_b][-i]
                res = int(
                    self.p.trial_runtime[self.cur_iter][self.cur_b][idx] /
                    small_job_sum_time)
                res = max(min(self.MAX_WORKER, self._num_worker, res), 1)

                for j in range(res):
                    gpu_idx = start_gpu % self._num_worker
                    gpu_dict[list(self.p._gpu_resources.keys())[gpu_idx]] = 1
                    start_gpu += 1
            else:
                gpu_idx = start_gpu % self._num_worker
                gpu_dict = {
                    list(self.p._gpu_resources.keys())[int(gpu_idx)]: 1
                }
                start_gpu += 1

            print("[Bin] Place trial {idx} on {gpu} : {res}".format(
                idx=trial.trial_id,
                res=res,
                gpu=list(self.p._gpu_resources.keys())[int(gpu_idx)]))

            self._launch_trial(trial, trial_runner, gpu_dict)
            logger.info("[Bin] Place trial {idx} on {gpu} : 1".format(
                idx=trial.trial_id,
                gpu=list(self.p._gpu_resources.keys())[int(gpu_idx)]))

    def _even_bin_packing(self, trial_runner):
        print("[Bin Packing]", self.p.base_rung_runtime_sum)
        avg_runtime = self.p.base_rung_runtime_sum[self.cur_iter][
            self.cur_b] / self.max_trial
        add_list = [idx for idx, val in
                    enumerate(self.p.trial_runtime[self.cur_iter][self.cur_b]) if val > avg_runtime]
        add_worker = min(len(add_list) * self.MAX_WORKER,
                         self._num_worker) - self.max_trial
        res_list = [1 for x in range(self.max_trial)]
        job_id = 0
        while (add_worker > 0):
            tmp_id = job_id % self.max_trial
            if tmp_id in add_list and res_list[tmp_id] < self.MAX_WORKER:
                res_list[tmp_id] += 1
                add_worker -= 1
            job_id += 1

        start_gpu = 0
        for idx, trial in enumerate(self.b.current_trials()):
            gpu_dict = {}
            if idx in add_list:
                for j in range(res_list[idx]):
                    assert start_gpu < self._num_worker
                    gpu_dict[list(self.p._gpu_resources.keys())[start_gpu]] = 1
                    start_gpu += 1
            else:
                assert start_gpu < self._num_worker
                gpu_dict = {list(self.p._gpu_resources.keys())[start_gpu]: 1}
                start_gpu += 1

            self._launch_trial(trial, trial_runner, gpu_dict)
            '''
            logger.info("[Bin] Place trial {idx} on {gpu} : 1".format(
                idx = trial.trial_id,
                gpu = list(self.p._gpu_resources.keys())[int(start_gpu)]) )
            '''
    def on_trial_result(self, trial_runner, trial, result) -> str:
        '''
        # check if this trial has finished current rung, if not, simply continue it
        if self.b.continue_trial(trial):
            return TrialScheduler.CONTINUE
        '''

        self.p.stage = 2
        # now the first trial has finished, transite to stage 2
        self.p._sm = Stage2(self.p, trial_runner)
        return self.p._sm.on_trial_result(trial_runner, trial, result)

    def on_trial_remove(self, trial_runner, trial):
        pass


class Stage2:
    def __init__(self, sched: FluidBandScheduler, trial_runner):
        print("[Stage2 init: Promoting jobs & Elastic]")
        self.p = sched

        # only supports run one bracket in one iteration for now
        self.cur_itr, self.cur_b, self.b = self.p.find_cur_braket()
        assert self.b is not None
        self.finished = [0 for x in range(self.b._s)]
        self.rung = 0
        if all([t.status == Trial.PAUSED for t in self.b.current_trials()]):
            self._promote(trial_runner)

        self.MAX_WORKER = 4  # HARDCODE optimal

    def _promote(self, trial_runner):

        promotables, bad = self.b.promotable(self.p._metric, self.p._metric_op)
        for t in promotables:
            print("[~~find promoted trial together]", t)
            self.b.promote(t)
            self.p._unpause_trial(trial_runner, t)
        # also cleanup any trials that need to be terminated
        for t in bad:
            if t.status == Trial.PAUSED:
                trial_runner.stop_trial(t)
            else:
                raise TuneError("Trial with unexpected status encountered")

    def _find_scale_trial(self, trials_running):
        print("@Find trials among {}".format(trials_running))

        runtime_time = [(self.p._res_estimator[t])[TIME_THIS_ITER_S]
                        for t in trials_running]

        def total_rung(s):
            return self.b._rungs[self.b._live_trials[s].rung].r

        remain_iter = [
            total_rung(t) - self.b._live_trials[t].result[TRAINING_ITERATION]
            for t in trials_running
        ]

        remaining = [a * b for a, b in zip(runtime_time, remain_iter)]

        print("Remaining time: {} s".format(remaining))
        if max(remaining) < 10:
            return None

        longest_trial = trials_running[np.argsort(remaining)[-1]]
        if len(longest_trial.resources.extra_custom_resources
               ) >= self.MAX_WORKER - 1:
            return None
        return longest_trial

    def _find_idle_worker(self):
        for idx, wk in enumerate(self.p.customized_gpu_wl):
            if not wk:
                return idx

        return None

    def on_trial_result(self, trial_runner, trial, result) -> str:
        print("stage2 on_trial_result: ", trial)
        if self._scale_up(trial_runner,
                          trial) and self.b.continue_trial(trial):
            print("stage2 scale up: ", trial)
            return TrialScheduler.PAUSE

        if self.b.continue_trial(trial):
            return TrialScheduler.CONTINUE

        # the trial has finished its current rung now,
        # by default we pause it to wait for others to finish
        # until something becomes promotable
        action = TrialScheduler.PAUSE
        # see if anything becomes promotable

        promotables, bad = self.b.promotable(self.p._metric, self.p._metric_op)

        for t in promotables:
            # print("[~~find promoted trial]", t)
            self.b.promote(t)
            if trial == t:
                assert t.status == trial.RUNNING
                action = TrialScheduler.CONTINUE
            else:
                # assert t.status == trial.PAUSED
                # t.update_resources(cpu=0, gpu=0,custom_resources = trial.resources.custom_resources)
                self.p._unpause_trial(trial_runner, t)
            logger.info(
                "[Early Promotion] Promote trial {id} with {res}".format(
                    id=t.trial_id, res=trial.resources))

        # if promotables is None:
        #     self._scale_up(trial_runner, trial)

        # also cleanup any trials that need to be terminated
        for t in self.b.cleanup_rungs():
            if trial == t:
                action = TrialScheduler.STOP
            else:
                trial_runner.stop_trial(t)

        # if nothing to promote
        # scale up the running trial with longest remaining time

        logger.info("{action} for {trial} on {metric}={metric_val}".format(
            action=action,
            trial=trial,
            metric=self.p._res_name,
            metric_val=result.get(self.p._res_name)))

        return action

    def _scale_up(self, trial_runner, trial):
        idle_wk_idx = self._find_idle_worker()  # only return one idle wk
        if idle_wk_idx is None:
            return None
        running_trials = []

        for wk in self.p.customized_gpu_wl:
            if wk:
                running_trials += list(wk)
        running_trials = list(dict.fromkeys(running_trials))

        num_runnnig = len(running_trials)
        if num_runnnig < 0:
            return None

        trial_to_scale_up = self._find_scale_trial(running_trials)
        if trial_to_scale_up is None or trial_to_scale_up != trial:
            return None
        new_res_dict = {}
        for k, v in trial_to_scale_up.resources.custom_resources.items():
            new_res_dict[k] = v

        increment_gpu = list(self.p._gpu_resources)[idle_wk_idx]
        new_res_dict[increment_gpu] = 1
        assert trial_to_scale_up.status == Trial.RUNNING, trial_to_scale_up.status
        # trial_runner.trial_executor.pause_trial(trial_to_scale_up)

        # self.p.update_resource( new_res_dict, trial_to_scale_up)
        self.p.update_res[trial_to_scale_up] = new_res_dict
        # assert trial_to_scale_up.status == Trial.PAUSED, trial_to_scale_up.status
        # trial_runner.trial_executor.unpause_trial( trial_to_scale_up)
        self.p.customized_gpu_wl[idle_wk_idx][trial_to_scale_up] = 1
        # assert trial_to_scale_up.status != Trial.PAUSED, trial_to_scale_up.status
        print("~~Try to scale up:", trial_to_scale_up, " to ", new_res_dict)

    def on_trial_remove(self, trial_runner, trial):
        self.finished[self.rung] += 1
        for idx, wk in enumerate(self.p.customized_gpu_wl):
            if trial in wk:
                del self.p.customized_gpu_wl[idx][trial]
                break
