from __future__ import annotations

import logging
import os
import random
import time
import copy
import traceback
from contextlib import contextmanager
import numpy as np
import itertools

import ray
from ray import ray_constants
from ray.resource_spec import ResourceSpec
from ray.exceptions import RayTimeoutError
from ray.tune.durable_trainable import DurableTrainable
from ray.tune.error import AbortTrialExecution, TuneError
from ray.tune.trial import Trial, Checkpoint, Location, TrialInfo
from ray.tune.trial_executor import TrialExecutor
from ray.tune.resources import Resources
from ray.tune.trainable import TrainableUtil
from ray.tune.result import TRIAL_INFO
from ray.tune.ray_trial_executor import _TrialCleanup, _to_gb, _LocalWrapper
from ray.tune.logger import NoopLogger
from ray.tune.error import TuneError
from ray.tune.result import TIME_THIS_ITER_S, TRAINING_ITERATION
from typing import NamedTuple

from .ray_custom_gpu_res import create_custom_gpu_res, gpu_idx_from_name
from .perf_manager import PerfManager

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import List, Optional, TypedDict, Tuple, Union, Dict, Any, Set, Iterable, TypeVar
    T = TypeVar('T')


logger = logging.getLogger(__name__)

DEFAULT_GET_TIMEOUT = 60.0  # seconds

class PendingJob(NamedTuple):
    trial: Trial
    checkpoint: Optional[Any]
    train: bool


class RunningJob(NamedTuple):
    trial: Trial
    in_flight_future: ray.ObjectID
    # committed resources
    resources: Resources


class TrialAndGroup(NamedTuple):
    trial: Trial
    group: int


class TrialGroupMeta:
    def __init__(self, grp: int, pendings: List[PendingJob]) -> None:
        self.grp = grp
        self.perf = PerfManager(trial_ids={p.trial.trial_id for p in pendings})
        self.trials = {
            p.trial.trial_id: p.trial
            for p in pendings
        }


class FluidExecutor(TrialExecutor):

    def __init__(self, **kwargs):
        super().__init__(queue_trials=True)  # type: ignore

        # resources
        self._avail_resources = Resources(cpu=0, gpu=0)
        self._committed_resources = Resources(cpu=0, gpu=0)
        self._resources_initialized = False
        self._last_resource_refresh = float("-inf")

        # make sure our own GPU resources are created first in the cluster
        create_custom_gpu_res()
        self._update_avail_resources()

        logger.info(f"Init with resources: {self._avail_resources}")

        self.jobs_pending: List[PendingJob] = []

        # map from in_flight_future to the job
        self.jobs_running: Dict[ray.ObjectID, RunningJob] = {}

        # used to save the previous run fut
        self.jobs_paused: Dict[ray.ObjectID, RunningJob] = {}

        # async queue to stop runner
        self._trial_cleanup = _TrialCleanup()

        # metadata about a trial group
        self.trial_group_meta: List[TrialGroupMeta] = []
        # trialgroup assignment,
        # mapping from trial_id to group num
        self.trial_groups: Dict[str, TrialAndGroup] = {}
    
    @property
    def num_trial_groups(self) -> int:
        return len(self.trial_group_meta)

    def _detect_groups(self):
        """Go over pending jobs, and assign trialgroup to them if not already done.
        If new groups are discovered, otherwise run static
        """
        logger.debug('_detect_groups')
        # pending may already be assigned a group if it's an unpaused trial
        assigned, unassigned = partition(self.jobs_pending, lambda p: p.trial.trial_id in self.trial_groups)
        unassigned = list(unassigned)
        assigned = list(assigned)
        if unassigned:
            meta = TrialGroupMeta(
                self.num_trial_groups,
                unassigned,
            )
            self.trial_group_meta.append(meta)
            logger.debug('Assign group %d to unassigned trials: %s', meta.grp, unassigned)
            for p in unassigned:
                self.trial_groups[p.trial.trial_id] = TrialAndGroup(p.trial, grp)
            # allocate reousrces
            self._static_fluid(meta)
        else:
            logger.debug('No new group')
        
        if assigned:
            # find each group with pending jobs and do dynamic
            groups = {
                self._find_group(p.trial.trial_id)
                for p in assigned
            }
            for meta in groups:
                self._dynamic_fluid(meta)
        else:
            logger.debug('No change in existing groups')
        
        # down with the pending, clear it
        self.jobs_pending.clear()

    def _dump_groups(self):
        """Dump group info for debugging"""
        logger.info("There are %d TrialGroup(s)", self.num_trial_groups)
        for grp in range(self.num_trial_groups):
            logger.info("TrialGroup %d", grp)
            for trial in self._trial_group(grp):
                if self._find_running(trial):
                    tag = 'jobs_running'
                elif self._find_pending(trial):
                    tag = 'jobs_pending'
                elif self._find_paused(trial):
                    tag = 'jobs_paused'
                else:
                    tag = 'none'
                logger.info("    Trial %s: [%s] queue [%s]", trial.trial_id, trial.status, tag)
        logger.info("Idle Resources: %s", self._resource_string(self.idle_resources))

    def _committed_resources_in_group(self, grp: int) -> Resources:
        """Compute all resources committed in this group"""
        used = Resources(cpu=0, gpu=0)
        for job in self.jobs_running.values():
            if job.trial.trial_id in self.trial_groups:
                used = resources_add(used, job.resources)
        return used

    def _static_fluid(self, meta: TrialGroupMeta):
        """Run static fluid on a specific group"""
        self._dump_groups()
        # set of trials to consider
        A = {trial.trial_id for trial in self._trial_group(meta.grp)}
        # assignment of resources
        W: Dict[str, Resources] = {}
        # compute new idle resources if every trials in this group were stopped
        M = resources_add(self.idle_resources, self._committed_resources_in_group(meta.grp))

        if meta.perf.trials_missing_info:
            # there are still trials need perf data,
            # restrict A to only these trials
            others = A.difference(meta.perf.trials_missing_info)
            A = meta.perf.trials_missing_info
            # set others to use 0 resource
            for tid in others:
                W[tid] = Resources(cpu=0, gpu=0)
            # use 1 gpu per trial to get reference perf
            for tid in A:
                r = Resources(cpu=1, gpu=1)
                Mp = Resources.subtract(M, r)
                if not Mp.is_nonnegative():
                    break
                M = Mp
                W[tid] = r
        else:
            # staticFluid
            # convert A to array for sorting
            A = np.array(list(A))
            # reference height (1 width)
            H1 = np.array([
                meta.perf.get_height(tid, 1)
                for tid in A
            ])
            # sort by H1 in non-increasing order
            ord = np.argsort(H1[::-1])
            A = A[ord]
            H1 = H1[ord]
            # $$w_i= \min(
            #   \max(
            #       \floor{
            #           \frac{h_{i,1}}{\sum_j h_{j,1} } n
            #       },
            #       \frac{1}{c}),
            #   d
            # )$$
            c = 1 / 2  # TODO: calc c
            d = 4  # TODO: calc d
            w = np.minimum(
                np.maximum(
                    np.floor(
                        H1 * np.size(H1) / np.sum(H1)
                    ),
                1 / c),
                d
            )
            # write to W
            W = dict(zip(A, w))

        self._ensure_W(W)

    def _dynamic_fluid(self, meta: TrialGroupMeta):
        """Re-compute and apply allocation changes"""
        self._dump_groups()

        # XXX: placeholder below
        temp = []
        for pending in self.jobs_pending:
            pending = self.jobs_pending.pop()

            if Resources.subtract(self.idle_resources, pending.trial.resources).is_nonnegative():
                self._kickoff(pending, pending.trial.resources)
            else:
                temp.append(pending)
        self.jobs_pending.extend(temp)
        # XXX: placeholder above

        # TODO: MPS & bin packing, modify pending/running queue

        # TODO: diff the pending and to_stop queue, modify them according to the plan above

        # TODO: step 2. for every running, adjust resource
        # TODO: step 3. call _kickoff for every pending, note that it may be None if failed to start
    
    def _ensure_W(self, W: np.array):
        """Adjust group resources given in W"""
        # stop any trials with 0 res
        # this has to be down first to free up resources for others to use
        # TODO: add to paused, then ensure_stop, we do not change trial's status which is visible outside
        running = None
        self.jobs_paused[running.in_flight_future] = running
        self._ensure_stop(running.trial)
        # TODO: adjust any trials with fewer res
        # TODO: kickoff any trials with >0 res, but not already running
        # use trial group to map trial_id to trial
        # construct PendingJob and use _kickoff to start the trial
        # TODO: adjust any trials with more res
        raise NotImplementedError

    def _find_group(self, trial: Trial) -> TrialGroupMeta:
        return self.trial_group_meta[self.trial_groups[trial.trial_id].group]

    def _trial_group(self, grp: int) -> List[Trial]:
        return [
            v.trial
            for v in self.trial_groups.values()
            if v.group == grp
        ]

    def _find_paused(self, trial) -> Optional[RunningJob]:
        for job in self.jobs_paused.values():
            if job.trial == trial:
                return job

    def _pop_paused(self, trial) -> Optional[RunningJob]:
        for fut, job in self.jobs_paused.items():
            if job.trial == trial:
                assert fut == job.in_flight_future
                return self.jobs_paused.pop(fut)

    def _find_running(self, trial) -> Optional[RunningJob]:
        for _, job in self.jobs_running.items():
            if job.trial == trial:
                return job
    
    def _find_pending(self, trial) -> Optional[PendingJob]:
        for job in self.jobs_pending:
            if job.trial == trial:
                return job

    def _setup_remote_runner(self, trial: Trial, res: Resources, reuse_allowed: bool) -> Any:
        trial.init_logger()
        # We checkpoint metadata here to try mitigating logdir duplication
        self.try_checkpoint_metadata(trial)
        remote_logdir = trial.logdir

        cls = ray.remote(
            num_cpus=res.cpu,
            num_gpus=res.gpu,
            memory=res.memory,
            object_store_memory=res.object_store_memory,
            resources=res.custom_resources)(
                trial.get_trainable_cls()
            )

        def logger_creator(config):
            # Set the working dir in the remote process, for user file writes
            os.makedirs(remote_logdir, exist_ok=True)
            if not ray.worker._mode() == ray.worker.LOCAL_MODE:
                os.chdir(remote_logdir)
            return NoopLogger(config, remote_logdir)

        # Clear the Trial's location (to be updated later on result)
        # since we don't know where the remote runner is placed.
        trial.set_location(Location())
        logger.debug("Trial %s: Setting up new remote runner.", trial)
        # Logging for trials is handled centrally by TrialRunner, so
        # configure the remote runner to use a noop-logger.
        trial_config = copy.deepcopy(trial.config)
        trial_config[TRIAL_INFO] = TrialInfo(trial)
        kwargs = {
            "config": trial_config,
            "logger_creator": logger_creator,
        }
        if issubclass(trial.get_trainable_cls(), DurableTrainable):
            kwargs["remote_checkpoint_dir"] = trial.remote_checkpoint_dir

        with _change_working_directory(trial):
            return cls.remote(**kwargs)

    def _kickoff(self, pending: PendingJob, res: Resources) -> Optional[RunningJob]:
        """Turn a pending job into a running one
        The pending job may be previously paused, or completely new.
        If paused, there will be a running job saved in the jobs_paused queue

        May return None if failed to start
        """
        trial = pending.trial
        self._commit_resources(res)

        # this is needed for the Trainer to setup distributed training
        # TODO: figure what config key is also needed to set resource info
        trial.resources = res

        try:
            reuse_allowed = pending.checkpoint is not None or trial.has_checkpoint()
            runner = self._setup_remote_runner(trial, res, reuse_allowed)
            trial.set_runner(runner)
            restore_job = self._restore(trial, pending.checkpoint)

            # trial's status is already RUNNING, set in start_trial, to fake a running trial from the outside

            # if previously is paused
            prev_run = self._pop_paused(trial)
            if prev_run is not None:
                if restore_job is not None:
                    logger.error("A previously paused job is restoring!!!, blocking on restoring")
                    ray.get(restore_job.in_flight_future)
                # add back to running queue
                self.jobs_running[prev_run.in_flight_future] = prev_run
                return prev_run

            # if is restoring
            if trial.is_restoring:
                assert restore_job is not None
                return restore_job

            # actually start train op
            return self._ensure_train(trial)
        except Exception as e:
            if isinstance(e, AbortTrialExecution):
                logger.exception("Trial %s: Error starting runner, aborting!", trial)
            else:
                logger.exception("Trial %s: Unexpected error starting runner.", trial)
            time.sleep(2)
            error_msg = traceback.format_exc()
            self._ensure_stop(
                trial, error=True, error_msg=error_msg, stop_logger=True,
                # NOTE that we don't return the resources, since they may have been lost.
                release_resources=False,
            )

    def _ensure_train(self, trial: Trial) -> RunningJob:
        """Actually invoke the train op on the runner"""
        assert trial.runner is not None
        with _change_working_directory(trial):
            fut = trial.runner.train.remote()

        if isinstance(fut, dict):
            # local mode
            fut = _LocalWrapper(fut)
        running = RunningJob(trial, fut)
        self.jobs_running[fut] = running
        return running

    def _ensure_stop(
        self,
        trial,
        error=False,
        error_msg='',
        stop_logger=True,
        release_resources=True
    ):
        """Stops the trial and its logger
        Handles any error
        """
        if stop_logger:
            trial.close_logger()
        
        prior_status = trial.status
        self.set_status(trial, Trial.ERROR if error else Trial.TERMINATED)
        trial.set_location(Location())

        # remove from running
        in_flight = [
            j
            for _, j in self.jobs_running.items()
            if j.trial == trial
        ]
        for j in in_flight:
            self.jobs_running.pop(j.in_flight_future)
            if release_resources:
                logger.debug("Trial %s: Returning resources.", trial)
                self._return_resources(j.resources)
        if in_flight:
            if prior_status not in [Trial.RUNNING, Trial.ERROR]:
                assert False, 'trial status invalid'

        # TODO: remove from trial group

        try:
            trial.write_error_log(error_msg)
            if hasattr(trial, "runner") and trial.runner:
                logger.debug("Trial %s: Destroying actor.", trial)
                with _change_working_directory(trial):
                    self._trial_cleanup.add(trial, actor=trial.runner)
        except Exception:
            logger.exception("Trial %s: Error stopping runner.", trial)
            self.set_status(trial, Trial.ERROR)
        finally:
            trial.set_runner(None)

    def has_resources(self, resources):
        """Tell the schedule algorithm to always submit trials to us"""
        return True
    
    def start_trial(self, trial, checkpoint=None, train=True):
        """Add to pending queue and reschedule"""
        logger.debug('start_trial %s', trial)
        # the trial is considered by the outside to be running
        self.set_status(trial, Trial.RUNNING)
        self.jobs_pending.append(PendingJob(trial, checkpoint, train))
        # The actual triggering is done in on_no_available_trials()

    def stop_trial(self, trial, error=False, error_msg=None, stop_logger=True):
        """Add to to-stop queue and reschedule"""
        logger.debug('stop_trial %s', trial)
        self._ensure_stop(trial, error, error_msg, stop_logger)
        meta = self._find_group(trial)
        self._dynamic_fluid(meta)

    def continue_training(self, trial):
        # this is called after got results from a trial,
        # and should start another train op in place of the
        # finished one.
        running_job = self._find_running(trial)
        if running_job is not None:
            # skip if the trial is running
            return

        # start new one
        self._ensure_train(trial)

    def pause_trial(self, trial):
        logger.debug('pause_trial %s', trial)
        running = self._find_running(trial)
        if running is not None:
            # add to jobs_paused
            self.jobs_paused[running.in_flight_future] = running
        # the super impl will call stop trial, which will then remove the job from running queue
        super().pause_trial(trial)
    
    def unpause_trial(self, trial):
        logger.debug('unpause_trial %s', trial)
        super().unpause_trial(trial)

    def resume_trial(self, trial):
        """Resumes PAUSED trials. This is a blocking call.
        This is not used by any algorithm
        """
        logger.debug('resume_trial %s', trial)
        assert trial.status == Trial.PAUSED, trial.status
        raise NotImplementedError

    def reset_trial(self, trial: Trial, new_config, new_experiment_tag):
        """Tries to invoke `Trainable.reset_config()` to reset trial.

        Args:
            trial (Trial): Trial to be reset.
            new_config (dict): New configuration for Trial trainable.
            new_experiment_tag (str): New experiment name for trial.

        Returns:
            True if `reset_config` is successful else False.
        """
        logger.debug('reset_trial %s', trial)
        trial.experiment_tag = new_experiment_tag
        trial.config = new_config
        trainable = trial.runner
        with _change_working_directory(trial):
            try:
                reset_val = ray.get(
                    trainable.reset_config.remote(new_config),
                    DEFAULT_GET_TIMEOUT
                )
            except RayTimeoutError:
                logger.exception("Trial %s: reset_config timed out.",
                                    trial)
                return False
        return reset_val
    
    def get_running_trials(self):
        return [job.trial for job in self.jobs_running.values()]
    
    def get_next_available_trial(self) -> Trial:
        """Return the next trial with ready result.
        Note that this doesn't remove the trial from running, fetch_result does that
        """
        futures = list(self.jobs_running.keys())
        # shuffle the list of futures because ray.wait
        # always return the first available future, but we want to be fair
        random.shuffle(futures)
        [ready_fut], _ = ray.wait(futures, num_returns=1)
        return self.jobs_running[ready_fut].trial
    
    def get_next_failed_trial(self) -> Optional[Trial]:
        if ray.worker._mode() == ray.worker.LOCAL_MODE:
            return None

        alive_node_ips = {
            node['NodeManagerAddress']
            for node in ray.state.nodes()
            if node['alive']
        }
        for trial in self.get_running_trials():
            if trial.node_ip and trial.node_ip not in alive_node_ips:
                return trial
        return None

    def fetch_result(self, trial):
        running_job = self._find_running(trial)
        assert running_job, "Trial was not running"
        self.jobs_running.pop(running_job.in_flight_future)
        result = ray.get(running_job.in_flight_future, DEFAULT_GET_TIMEOUT)
        if isinstance(result, _LocalWrapper):
            result = result.unwrap()

        # notify trial group
        meta = self._find_group(trial)
        meta.perf.on_trial_result(trial.trial_id, result)
        return result

    def debug_string(self):
        # TODO debug_string
        pass

    def _resource_string(self, res: Resources) -> str:
        """Returns a string describing the total resources available."""
        res_str = (
            f"{res.cpu} CPUs, {res.gpu} GPUs, "
            f"{_to_gb(res.memory)} GiB heap, "
            f"{_to_gb(res.object_store_memory)} GiB objects"
        )
        if res.custom_resources:
            custom = ", ".join(
                f"{res.get_res_total(name)} {name}"
                for name in res.custom_resources
            )
            res_str += f" ({custom})"
        return res_str

    def save(self, trial, storage=Checkpoint.PERSISTENT, result=None):
        """Saves the trial's state to a checkpoint asynchronously.

        Args:
            trial (Trial): The trial to be saved.
            storage (str): Where to store the checkpoint. Defaults to
                PERSISTENT.
            result (dict): The state of this trial as a dictionary to be saved.
                If result is None, the trial's last result will be used.

        Returns:
             Checkpoint object, or None if an Exception occurs.
        """
        result = result or trial.last_result
        with _change_working_directory(trial):
            if storage == Checkpoint.MEMORY:
                value = trial.runner.save_to_object.remote()
                checkpoint = Checkpoint(storage, value, result)
                trial.on_checkpoint(checkpoint)
            else:
                value = trial.runner.save.remote()
                checkpoint = Checkpoint(storage, value, result)
                trial.saving_to = checkpoint
                self.jobs_running[value] = RunningJob(trial, value)
        return checkpoint

    def _restore(self, trial, checkpoint=None, block=False) -> Optional[RunningJob]:
        """Restores training state from a given model checkpoint.

        Args:
            trial (Trial): The trial to be restored.
            checkpoint (Checkpoint): The checkpoint to restore from. If None,
                the most recent PERSISTENT checkpoint is used. Defaults to
                None.
            block (bool): Whether or not to block on restore before returning.

        Raises:
            RuntimeError: This error is raised if no runner is found.
            AbortTrialExecution: This error is raised if the trial is
                ineligible for restoration, given the Tune input arguments.
        """
        if checkpoint is None or checkpoint.value is None:
            checkpoint = trial.checkpoint
        if checkpoint.value is None:
            return
        if trial.runner is None:
            raise RuntimeError(
                "Trial {}: Unable to restore - no runner found.".format(trial))
        value = checkpoint.value
        if checkpoint.storage == Checkpoint.MEMORY:
            logger.debug("Trial %s: Attempting restore from object", trial)
            # Note that we don't store the remote since in-memory checkpoints
            # don't guarantee fault tolerance and don't need to be waited on.
            with _change_working_directory(trial):
                trial.runner.restore_from_object.remote(value)
        else:
            logger.debug("Trial %s: Attempting restore from %s", trial, value)
            if issubclass(trial.get_trainable_cls(), DurableTrainable):
                with _change_working_directory(trial):
                    remote = trial.runner.restore.remote(value)
            elif trial.sync_on_checkpoint:
                # This provides FT backwards compatibility in the
                # case where a DurableTrainable is not provided.
                logger.warning("Trial %s: Reading checkpoint into memory.",
                               trial)
                data_dict = TrainableUtil.pickle_checkpoint(value)
                with _change_working_directory(trial):
                    remote = trial.runner.restore_from_object.remote(data_dict)
            else:
                raise AbortTrialExecution(
                    "Pass in `sync_on_checkpoint=True` for driver-based trial"
                    "restoration. Pass in an `upload_dir` and a Trainable "
                    "extending `DurableTrainable` for remote storage-based "
                    "restoration")

            if block:
                ray.get(remote)
            else:
                trial.restoring_from = checkpoint
                running_job = RunningJob(trial, remote)
                self.jobs_running[remote] = running_job
                return running_job

    def restore(self, trial, checkpoint=None, block=False):
        return self._restore(trial, checkpoint, block)

    def export_trial_if_needed(self, trial: Trial):
        """Exports model of this trial based on trial.export_formats.

        Return:
            A dict that maps ExportFormats to successfully exported models.
        """
        if trial.export_formats and len(trial.export_formats) > 0:
            with _change_working_directory(trial):
                return ray.get(
                    trial.runner.export_model.remote(trial.export_formats),
                    DEFAULT_GET_TIMEOUT
                )
        return {}

    def cleanup(self):
        self._trial_cleanup.cleanup(partial=False)

    def has_gpus(self):
        if not self._resources_initialized:
            self._update_avail_resources()
        return self._avail_resources.gpu > 0

    def on_step_begin(self, trial_runner):
        """Before step() called, update the available resources."""
        self._update_avail_resources()

    def _update_avail_resources(self, num_retries=5):
        resources = None
        for i in range(num_retries):
            if i > 0:
                logger.warning(
                    "Cluster resources not detected or are 0. Attempt #"
                    "%s...", i + 1)
                time.sleep(0.5)
            try:
                resources = ray.cluster_resources()
            except Exception:
                # TODO(rliaw): Remove this when local mode is fixed.
                # https://github.com/ray-project/ray/issues/4147
                logger.debug("Using resources for local machine.")
                resources = ResourceSpec().resolve(True).to_resource_dict()
            if resources:
                break

        if not resources:
            # NOTE: This hides the possibility that Ray may be waiting for
            # clients to connect.
            resources.setdefault("CPU", 0)
            resources.setdefault("GPU", 0)
            logger.warning("Cluster resources cannot be detected or are 0. "
                           "You can resume this experiment by passing in "
                           "`resume=True` to `run`.")

        resources = resources.copy()
        num_cpus = resources.pop("CPU", 0)
        num_gpus = resources.pop("GPU", 0)
        memory = ray_constants.from_memory_units(resources.pop("memory", 0))
        object_store_memory = ray_constants.from_memory_units(
            resources.pop("object_store_memory", 0))
        custom_resources = resources

        avail_resources = Resources(
            int(num_cpus),
            int(num_gpus),
            memory=int(memory),
            object_store_memory=int(object_store_memory),
            custom_resources=custom_resources)
        
        assert self.idle_resources.is_nonnegative(), "Cluster removed resources from running trials!"

        self._avail_resources = avail_resources
        self._last_resource_refresh = time.time()
        self._resources_initialized = True

    @property
    def idle_resources(self) -> Resources:
        return Resources.subtract(self._avail_resources, self._committed_resources)

    def _commit_resources(self, resources):
        committed = self._committed_resources
        all_keys = set(resources.custom_resources).union(
            set(committed.custom_resources))

        custom_resources = {
            k: committed.get(k) + resources.get_res_total(k)
            for k in all_keys
        }

        self._committed_resources = Resources(
            committed.cpu + resources.cpu_total(),
            committed.gpu + resources.gpu_total(),
            committed.memory + resources.memory_total(),
            committed.object_store_memory +
            resources.object_store_memory_total(),
            custom_resources=custom_resources)

    def _return_resources(self, resources):
        committed = self._committed_resources

        all_keys = set(resources.custom_resources).union(
            set(committed.custom_resources))

        custom_resources = {
            k: committed.get(k) - resources.get_res_total(k)
            for k in all_keys
        }
        self._committed_resources = Resources(
            committed.cpu - resources.cpu_total(),
            committed.gpu - resources.gpu_total(),
            custom_resources=custom_resources)

        assert self._committed_resources.is_nonnegative(), (
            "Resource invalid: {}".format(resources))
    
    def on_no_available_trials(self, trial_runner):
        """This is called when we get all trial from a batch from the search algo"""
        logger.debug('on_no_available_trials')
        self._detect_groups()
        super().on_no_available_trials(trial_runner)


@contextmanager
def _change_working_directory(trial: Trial):
    """Context manager changing working directory to trial logdir.
    Used in local mode.

    For non-local mode it is no-op.
    """
    if ray.worker._mode() == ray.worker.LOCAL_MODE:
        old_dir = os.getcwd()
        try:
            os.chdir(trial.logdir)
            yield
        finally:
            os.chdir(old_dir)
    else:
        yield


def resources_add(a: Resources, b: Resources) -> Resources:
    zero = Resources(cpu=0, gpu=0)
    nb = Resources.subtract(zero, b)
    return Resources.subtract(a, nb)


def partition(iterable: Iterable[T], pred) -> Tuple[Iterable[T], Iterable[T]]:
    """Partition an iterable into two with given pred"""
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)
