import logging
import numbers
import os
import random
import tempfile
import time

import numpy as np
import ray
import torch
from ray.exceptions import RayActorError
from ray.tune import Trainable
from ray.tune.utils.util import merge_dicts
from ray.util.sgd.torch.constants import VALID_SCHEDULER_STEP
from ray.util.sgd.torch.distributed_torch_runner import (
    DeactivatedRunner,
    DistributedTorchRunner,
    LocalDistributedRunner,
)
from ray.util.sgd.torch.torch_runner import TorchRunner
from ray.util.sgd.utils import BATCH_SIZE, NUM_SAMPLES, check_for_failure

logger = logging.getLogger(__name__)
RESIZE_COOLDOWN_S = 10


def _validate_scheduler_step_freq(scheduler_step_freq):
    """This validation check only happens if a scheduler is passed in."""
    if scheduler_step_freq not in VALID_SCHEDULER_STEP:
        raise ValueError(
            "Scheduler step freq must be in {}. Got {}".format(
                VALID_SCHEDULER_STEP, scheduler_step_freq
            )
        )


def _remind_gpu_usage(use_gpu):
    if not use_gpu and torch.cuda.is_available():
        logger.info(
            "GPUs detected but not using them. Set `use_gpu` to " "enable GPU usage. "
        )


# XXX: currently, we allocate both GPU and a custom resource unique per GPU to each actor.
# This way, each actor actually gets GPU automatically and placement is enforced by the custom
# resource.
# However, when one physical node has multiple GPUs, this may break. Consider the following case:
#     actor A: gpu=0.4, fluid_GPU_0_abcde=0.4
#     actor B: gpu=0.4, fluid_GPU_1_abcde=0.4
# Both actor will be placed on host abcde, however, they will likely get the same GPU assigned
# regard less they have different custom resource assigend. The CUDA_VISIABLE_DEVICES variable
# may be incorrect in this case.
def _set_device_from_fluid_res():

    for node in ray.nodes():
        num_gpus = node["Resources"].get("GPU", 0)
        if num_gpus == 0:
            return

    for k, _ in ray.get_resource_ids().items():
        if not k.startswith("fluid_GPU_"):
            continue
        gpu_id = int(k.split("_")[2])
        torch.cuda.set_device(gpu_id)
        break


class TorchTrainer:
    """Train a PyTorch model using distributed PyTorch.

    Launches a set of actors which connect via distributed PyTorch and
    coordinate gradient updates to train the provided model. If Ray is not
    initialized, TorchTrainer will automatically initialize a local Ray
    cluster for you. Be sure to run `ray.init(address="auto")` to leverage
    multi-node training.

    .. code-block:: python

        def model_creator(config):
            return nn.Linear(1, 1)


        def optimizer_creator(model, config):
            return torch.optim.SGD(
                model.parameters(), lr=config.get("lr", 1e-4))


        def data_creator(config):
            batch_size = config["batch_size"]
            train_data, val_data = LinearDataset(2, 5), LinearDataset(2, 5)
            train_loader = DataLoader(train_data, batch_size=batch_size)
            val_loader = DataLoader(val_data, batch_size=batch_size)
            return train_loader, val_loader


        trainer = TorchTrainer(
            model_creator=model_creator,
            data_creator=data_creator,
            optimizer_creator=optimizer_creator,
            loss_creator=nn.MSELoss,
            config={"batch_size": 32},
            use_gpu=True
        )
        for i in range(4):
            trainer.train()

    The creator functions will execute before distributed coordination and
    training is setup. This is so that creator functions that download
    large datasets will not trigger any timeouts.

    The order of operations for creator functions are:

    ``data_creator`` -> ``model_creator`` -> ``optimizer_creator`` ->
    ``scheduler_creator`` -> ``loss_creator``.

    Args:
        model_creator (dict -> Model(s)): Constructor function that takes in
            config and returns the model(s) to be optimized. These must be
            ``torch.nn.Module`` objects. If multiple models are returned,
            a ``training_operator_cls`` must be specified. You do not need to
            handle GPU/devices in this function; RaySGD will do that under
            the hood.
        data_creator (dict -> Iterable(s)): Constructor function
            that takes in the passed config and returns one or
            two Iterable objects. Note that even though two Iterable objects
            can be returned, only one will be used for training, and the
            other will be used for validation. If not provided, you must
            provide a custom TrainingOperator.
        optimizer_creator ((models, dict) -> optimizers): Constructor
            function that takes in the return values from
            ``model_creator`` and the passed config and returns One or
            more Torch optimizer objects. You do not need to handle
            GPU/devices in this function; ``RaySGD`` will do that for you.
        loss_creator (torch.nn.*Loss class | dict -> loss): A constructor
            function for the training loss. This can be either a function that
            takes in the provided config for customization or a subclass
            of ``torch.nn.modules.loss._Loss``, which is most Pytorch
            loss classes. For example, ``loss_creator=torch.nn.BCELoss``.
            If not provided, you must provide a custom TrainingOperator.
        scheduler_creator ((optimizers, dict) -> scheduler):
            A constructor function for the torch scheduler. This is
            a function that takes in the generated optimizers (from
            ``optimizer_creator``) provided config for customization.
            Be sure to set ``scheduler_step_freq`` to increment the
            scheduler correctly.
        training_operator_cls (type): Custom training operator class
            that subclasses the TrainingOperator class. This class
            will be copied onto all remote workers and used to specify
            custom training and validation operations. Defaults to
            TrainingOperator.
        config (dict): Custom configuration value to be passed to
            all creator and operator constructors.
        num_workers (int): the number of workers used in distributed
            training. If 1, the worker will not be wrapped with
            DistributedDataParallel.
        use_gpu (bool): Sets resource allocation for workers to 1 GPU
            if true, and automatically moves both the model and optimizer
            to the available CUDA device.
        backend (string): backend used by distributed PyTorch. Currently
            support "nccl", "gloo", and "auto". If "auto", RaySGD will
            automatically use "nccl" if `use_gpu` is True, and "gloo"
            otherwise.
        serialize_data_creation (bool): A filelock will be used
            to ensure no race conditions in data downloading among
            different workers on the same node (using the local file system).
            Defaults to True.
        wrap_ddp (bool): Whether to automatically wrap DistributedDataParallel
            over each model. If False, you are expected to call it yourself.
        add_dist_sampler (bool): Whether to automatically add a
            DistributedSampler to all created dataloaders. Only applicable
            if num_workers > 1.
        use_fp16 (bool): Enables mixed precision training via apex if apex
            is installed. This is automatically done after the model and
            optimizers are constructed and will work for multi-model training.
            Please see https://github.com/NVIDIA/apex for more details.
        apex_args (dict|None): Dict containing keyword args for amp.initialize.
            See https://nvidia.github.io/apex/amp.html#module-apex.amp. By
            default, the models and optimizers are passed in. Consider using
            "num_losses" if operating over multiple models and optimizers.
        scheduler_step_freq: "batch", "epoch", "manual", or None. This will
            determine when ``scheduler.step`` is called. If "batch",
            ``step`` will be called after every optimizer step. If "epoch",
            ``step`` will be called after one pass of the DataLoader. If
            "manual", the scheduler will not be incremented automatically -
            you are expected to call ``trainer.update_schedulers`` manually.
            If a scheduler is passed in, this value is expected to not be None.

    """

    def __init__(
        self,
        *,
        model_creator,
        data_creator,
        optimizer_creator,
        loss_creator=None,
        scheduler_creator=None,
        training_operator_cls=None,
        initialization_hook=None,
        config=None,
        use_gpu="auto",
        backend="auto",
        wrap_ddp=True,
        serialize_data_creation=True,
        use_fp16=False,
        use_tqdm=False,
        apex_args=None,
        add_dist_sampler=True,
        scheduler_step_freq=None,
    ):
        if not (
            callable(model_creator)
            and callable(optimizer_creator)
            and callable(data_creator)
        ):
            raise ValueError(
                "Must provide a callable model_creator, optimizer_creator, "
                "and data_creator."
            )

        self.model_creator = model_creator
        self.optimizer_creator = optimizer_creator
        self.loss_creator = loss_creator
        self.data_creator = data_creator
        self.scheduler_creator = scheduler_creator
        self.training_operator_cls = training_operator_cls

        if not training_operator_cls and not loss_creator:
            raise ValueError(
                "If a loss_creator is not provided, you must "
                "provide a custom training operator."
            )

        self.initialization_hook = initialization_hook
        self.config = {} if config is None else config
        if use_gpu == "auto":
            use_gpu = torch.cuda.is_available()

        _remind_gpu_usage(use_gpu)

        num_workers = self._check_potential_workers_size()

        if backend == "auto":
            backend = "nccl" if use_gpu else "gloo"

        logger.debug("Using {} as backend.".format(backend))
        self.backend = backend
        self.use_gpu = use_gpu

        self.serialize_data_creation = serialize_data_creation
        self.wrap_ddp = wrap_ddp
        self.use_fp16 = use_fp16
        self.use_tqdm = use_tqdm
        self.add_dist_sampler = add_dist_sampler

        if apex_args and not isinstance(apex_args, dict):
            raise ValueError("apex_args needs to be a dict object.")

        self.apex_args = apex_args
        self.temp_dir = tempfile.mkdtemp(prefix="raysgd")
        self._num_failures = 0
        self._last_resize = float("-inf")

        self.local_worker = DeactivatedRunner()
        self.remote_workers = []

        if scheduler_creator:
            _validate_scheduler_step_freq(scheduler_step_freq)

        self.scheduler_step_freq = scheduler_step_freq

        if not ray.is_initialized() and num_workers > 1:
            logger.info(
                "Automatically initializing single-node Ray. To use "
                "multi-node training, be sure to run `ray.init("
                "address='auto')` before instantiating the Trainer."
            )
            ray.init()
        self._start_workers(num_workers)

    def _configure_and_split_batch(self, num_workers):
        """If sgd.utils.BATCH_SIZE is provided, split among workers."""
        if BATCH_SIZE not in self.config:
            return
        # Compute batch size per worker
        logger.debug("BATCH_SIZE parameter detected. Splitting among workers.")
        batch_size = self.config[BATCH_SIZE]
        batch_size_per_worker = batch_size // num_workers
        if batch_size % num_workers > 0:
            new_batch_size = batch_size_per_worker * num_workers
            logger.warning(
                (
                    "Changing batch size from {old_batch_size} to "
                    "{new_batch_size} to evenly distribute batches across "
                    "{num_workers} workers."
                ).format(
                    old_batch_size=batch_size,
                    new_batch_size=new_batch_size,
                    num_workers=num_workers,
                )
            )
            self.config[BATCH_SIZE] = new_batch_size
        return batch_size_per_worker

    def _start_workers(self, num_workers):
        logger.debug(f"start_workers: Setting {num_workers} workers.")
        worker_config = self.config.copy()
        batch_size_per_worker = self._configure_and_split_batch(num_workers)
        if batch_size_per_worker:
            worker_config[BATCH_SIZE] = batch_size_per_worker

        params = dict(
            model_creator=self.model_creator,
            data_creator=self.data_creator,
            optimizer_creator=self.optimizer_creator,
            loss_creator=self.loss_creator,
            scheduler_creator=self.scheduler_creator,
            training_operator_cls=self.training_operator_cls,
            config=worker_config,
            use_fp16=self.use_fp16,
            use_gpu=True,
            use_tqdm=self.use_tqdm,
            apex_args=self.apex_args,
            scheduler_step_freq=self.scheduler_step_freq,
        )

        if num_workers == 1:
            # Start local worker
            self.local_worker = TorchRunner(**params)
            self.apply_all_workers(_set_device_from_fluid_res)
            if self.initialization_hook:
                self.apply_all_workers(self.initialization_hook)
            self.local_worker.setup()
        else:
            params.update(
                backend=self.backend,
                add_dist_sampler=self.add_dist_sampler,
                wrap_ddp=self.wrap_ddp,
            )

            # Start local worker
            self.local_worker = LocalDistributedRunner(**params)

            # Start remote workers
            # assert num_workers == len(self.extra_assigned_worker_res) + 1
            self.remote_workers = []
            for res_name, res_val in self.extra_assigned_worker_res:
                # Generate actor class
                RemoteRunner = ray.remote(
                    num_cpus=1, num_gpus=res_val, resources={res_name: res_val}
                )(DistributedTorchRunner)
                self.remote_workers.append(RemoteRunner.remote(**params))

            self.apply_all_workers(_set_device_from_fluid_res)
            if self.initialization_hook:
                self.apply_all_workers(self.initialization_hook)

            # Compute URL for initializing distributed PyTorch
            ip = ray.services.get_node_ip_address()
            port = self.local_worker.find_free_port()

            address = "tcp://{ip}:{port}".format(ip=ip, port=port)

            # Runs the creator functions.
            remote_component_setup = [
                worker.setup_components.remote()
                for i, worker in enumerate(self.remote_workers)
            ]
            self.local_worker.setup_components()
            # Get setup tasks in order to throw errors on failure
            ray.get(remote_component_setup)

            # Setup the process group among all workers.
            remote_pgroup_setups = [
                worker.setup_process_group.remote(address, i + 1, num_workers)
                for i, worker in enumerate(self.remote_workers)
            ]
            self.local_worker.setup_process_group(address, 0, num_workers)
            # Get setup tasks in order to throw errors on failure
            ray.get(remote_pgroup_setups)

            # Runs code that requires all creator functions to have run.
            remote_operator_setups = [
                worker.setup_ddp_and_operator.remote() for worker in self.remote_workers
            ]
            self.local_worker.setup_ddp_and_operator()
            # Get setup tasks in order to throw errors on failure
            ray.get(remote_operator_setups)

    def train(
        self,
        num_steps=None,
        profile=False,
        reduce_results=True,
        max_retries=3,
        info=None,
    ):
        """Runs a training epoch.

        Calls `operator.train_epoch()` on N parallel workers simultaneously
        underneath the hood.

        Set `max_retries` to enable fault handling in case of
        instance preemption.

        Args:
            num_steps (int): Number of batches to compute update steps on.
                This corresponds also to the number of times
                ``TrainingOperator.train_batch`` is called.
            profile (bool): Returns time stats for the training procedure.
            reduce_results (bool): Whether to average all metrics across
                all workers into one dict. If a metric is a non-numerical
                value (or nested dictionaries), one value will be randomly
                selected among the workers. If False, returns a list of dicts.
            max_retries (int): Must be non-negative. If set to N, TorchTrainer
                will detect and recover from training failure. The recovery
                process will kill all current workers, query the Ray
                global state for total available resources, and re-launch up to
                the available resources. Behavior is not well-defined
                in case of shared cluster usage. Defaults to 3.
            info (dict): Optional dictionary passed to the training
                operator for ``train_epoch`` and ``train_batch``.

        Returns:
            (dict | list) A dictionary of metrics for training.
                You can provide custom metrics by passing in a custom
                ``training_operator_cls``. If ``reduce_results=False``,
                this will return a list of metric dictionaries whose
                length will be equal to ``num_workers``.
        """
        assert max_retries >= 0, "`max_retries` must be non-negative."
        if self._should_resize():
            logger.info("Resize opportunity detected. Attempting to scale up.")
            self._resize_workers()
        success, worker_stats = self._train_epoch(
            num_steps=num_steps, profile=profile, info=info
        )
        # Fault handling
        for i in range(max_retries):
            if success:
                break
            else:
                self._num_failures += 1
            self._resize_workers()
            logger.info(
                "Retrying training step with %d workers."
                % (len(self.remote_workers) + 1)
            )
            success, worker_stats = self._train_epoch(
                num_steps=num_steps, profile=profile, info=info
            )
        if not success:
            raise RuntimeError("Training run failed.")

        if reduce_results:
            return self._process_stats(worker_stats)
        else:
            return worker_stats

    def _process_stats(self, worker_stats):
        stats = {
            NUM_SAMPLES: sum(stats.pop(NUM_SAMPLES, np.nan) for stats in worker_stats)
        }

        for stat_key in worker_stats[0]:
            if isinstance(worker_stats[0], numbers.Number):
                stats[stat_key] = np.nanmean(
                    [s.get(stat_key, np.nan) for s in worker_stats]
                )
            else:
                stats[stat_key] = worker_stats[0][stat_key]
        return stats

    def _train_epoch(self, num_steps=None, profile=False, info=None):
        params = dict(num_steps=num_steps, profile=profile, info=info)

        remote_worker_stats = [
            w.train_epoch.remote(**params) for w in self.remote_workers
        ]

        try:
            local_worker_stats = self.local_worker.train_epoch(**params)
        except RuntimeError as err:
            if "gloo" in err.args[0] and "Timed out" in err.args[0]:
                logger.warning(err)
                return False, None
            if "NCCL" in err.args[0]:  # there is no specific error message
                logger.warning(err)
                return False, None

            raise err

        success = check_for_failure(remote_worker_stats)
        if success:
            return success, [local_worker_stats] + ray.get(remote_worker_stats)

        return success, None

    def apply_all_workers(self, fn):
        """Run a function on all operators on the workers.

        Args:
            fn (Callable): A function that takes in no arguments.

        Returns:
            A list of objects returned by ``fn`` on each worker.

        """
        remote_calls = [w.apply.remote(fn) for w in self.remote_workers]
        local_call = self.local_worker.apply(fn)
        return [local_call] + ray.get(remote_calls)

    def apply_all_operators(self, fn):
        """Run a function on all operators on the workers.

        Args:
            fn (Callable[TrainingOperator]): A function that takes in a
                TrainingOperator.

        Returns:
            A list of objects returned by ``fn`` on each operator.

        """
        remote_calls = [w.apply_operator.remote(fn) for w in self.remote_workers]
        local_call = self.local_worker.apply_operator(fn)
        return [local_call] + ray.get(remote_calls)

    def validate(self, num_steps=None, profile=False, reduce_results=True, info=None):
        """Evaluates the model on the validation data set.

        Args:
            num_steps (int): Number of batches to compute update steps on.
                This corresponds also to the number of times
                ``TrainingOperator.validate_batch`` is called.
            profile (bool): Returns time stats for the evaluation procedure.
            reduce_results (bool): Whether to average all metrics across
                all workers into one dict. If a metric is a non-numerical
                value (or nested dictionaries), one value will be randomly
                selected among the workers. If False, returns a list of dicts.
            info (dict): Optional dictionary passed to the training
                operator for `validate` and `validate_batch`.

        Returns:
            A dictionary of metrics for validation.
                You can provide custom metrics by passing in a custom
                ``training_operator_cls``.
        """
        params = dict(num_steps=num_steps, profile=profile, info=info)

        remote_worker_stats = [w.validate.remote(**params) for w in self.remote_workers]
        local_worker_stats = self.local_worker.validate(**params)
        worker_stats = [local_worker_stats] + ray.get(remote_worker_stats)

        if reduce_results:
            return self._process_stats(worker_stats)
        else:
            return worker_stats

    def update_scheduler(self, metric):
        """Calls ``scheduler.step(metric)`` on all schedulers.

        This is useful for lr_schedulers such as ``ReduceLROnPlateau``.
        """
        self.apply_all_operators(
            lambda op: [sched.step(metric) for sched in op.schedulers]
        )

    def get_model(self):
        """Returns the learned model(s)."""
        unwrapped = []
        for model in self.local_worker.models:
            unwrapped += [model.module if hasattr(model, "module") else model]
        if len(unwrapped) == 1:
            return unwrapped[0]
        return unwrapped

    def get_local_operator(self):
        """Returns the local TrainingOperator object.

        Be careful not to perturb its state, or else you can cause the system
        to enter an inconsistent state.

        Returns:
            TrainingOperator: The local TrainingOperator object.
        """
        return self.local_worker.training_operator

    def state_dict(self):
        return self.local_worker.state_dict()

    def load_state_dict(self, state_dict, blocking=False):
        # This is not the most efficient because you have to wait for
        # the local worker to save then dump to buffer.
        self.local_worker.load_state_dict(state_dict)
        state_id = ray.put(self.local_worker.state_stream())

        remote_calls = [
            worker.load_state_stream.remote(state_id) for worker in self.remote_workers
        ]
        if blocking:
            ray.get(remote_calls)

    def save(self, checkpoint):
        """Saves the Trainer state to the provided checkpoint path.

        Args:
            checkpoint (str): Path to target checkpoint file.
        """
        torch.save(self.state_dict(), checkpoint)
        return checkpoint

    def load(self, checkpoint):
        """Loads the Trainer and all workers from the provided checkpoint.

        Args:
            checkpoint (str): Path to target checkpoint file.
        """
        state_dict = torch.load(checkpoint)
        self.load_state_dict(state_dict)

    def restore(self, *args):
        raise DeprecationWarning("Use `TorchTrainer.load()` instead.")

    def shutdown(self, force=False):
        """Shuts down workers and releases resources."""
        if not force:
            cleanup = [worker.shutdown.remote() for worker in self.remote_workers]
            self.local_worker.shutdown()
            try:
                ray.get(cleanup)
                [worker.__ray_terminate__.remote() for worker in self.remote_workers]
            except RayActorError:
                logger.warning("Failed to shutdown gracefully, forcing a shutdown.")

                for worker in self.remote_workers:
                    logger.warning("Killing worker {}.".format(worker))
                    ray.kill(worker)
        else:
            self.local_worker.shutdown()
            for worker in self.remote_workers:
                logger.debug("Killing worker {}.".format(worker))
                ray.kill(worker)

        self.local_worker = DeactivatedRunner()
        self.remote_workers = []

    def _reset(self):
        """Terminates models without giving up local resource reservation."""
        self.local_worker.shutdown(cleanup=False)
        for worker in self.remote_workers:
            logger.debug("Killing worker {}.".format(worker))
            ray.kill(worker)
        self.local_worker = DeactivatedRunner()
        self.remote_workers = []

    @property
    def extra_assigned_worker_res(self):
        return list(self.config.get("extra_fluid_trial_resources", {}).items())

    def _check_potential_workers_size(self):
        # ASSUME 1 GPU + 1 CPU is already reserved for the local worker
        num_workers = len(self.extra_assigned_worker_res) + 1  # including self
        # print("**number of worker: ", num_workers)
        # print("extra_assigned_worker_res: ", self.extra_assigned_worker_res)

        return num_workers

    def _resize_workers(self, max_retries=10):
        self._reset()

        time.sleep(1)
        for i in range(max_retries):
            num_workers = self._check_potential_workers_size()
            if num_workers:
                self._last_resize = time.time()
                self._start_workers(num_workers)
                self.load_state_dict(self.state_dict())
                return
        raise RuntimeError("Exceeded max_retries for relaunching workers.")

    def _should_resize(self):
        """Returns True if past cooldown and exists resources to scale up."""
        return len(self.remote_workers) + 1 != self._check_potential_workers_size()

    @classmethod
    def as_trainable(cls, *args, **kwargs):
        """Creates a BaseTorchTrainable class compatible with Tune.

        Any configuration parameters will be overriden by the Tune
        Trial configuration. You can also subclass the provided Trainable
        to implement your own iterative optimization routine.

        .. code-block:: python

            TorchTrainable = TorchTrainer.as_trainable(
                model_creator=ResNet18,
                data_creator=cifar_creator,
                optimizer_creator=optimizer_creator,
                loss_creator=nn.CrossEntropyLoss,
                num_gpus=2
            )
            analysis = tune.run(
                TorchTrainable,
                config={"lr": tune.grid_search([0.01, 0.1])}
            )

        """

        class TorchTrainable(BaseTorchTrainable):
            def _create_trainer(self, tune_config):
                """Overrides the provided config with Tune config."""
                provided_config = kwargs.get("config", {}).copy()
                provided_config.update(tune_config)
                kwargs["config"] = provided_config
                trainer = TorchTrainer(*args, **kwargs)
                return trainer

        return TorchTrainable


class BaseTorchTrainable(Trainable):
    """Base class for converting TorchTrainer to a Trainable class.

    This class is produced when you call ``TorchTrainer.as_trainable(...)``.

    You can override the produced Trainable to implement custom iterative
    training procedures:

    .. code-block:: python

        TorchTrainable = TorchTrainer.as_trainable(
            model_creator=ResNet18,
            data_creator=cifar_creator,
            optimizer_creator=optimizer_creator,
            loss_creator=nn.CrossEntropyLoss,
            num_gpus=2
        )
        # TorchTrainable is subclass of BaseTorchTrainable.

        class CustomTrainable(TorchTrainable):
            def _train(self):
                for i in range(5):
                    train_stats = self.trainer.train()
                validation_stats = self.trainer.validate()
                train_stats.update(validation_stats)
                return train_stats

        analysis = tune.run(
            CustomTrainable,
            config={"lr": tune.grid_search([0.01, 0.1])}
        )

    """

    def _setup(self, config):
        """Constructs a TorchTrainer object as `self.trainer`."""
        seed = config.get("seed")
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        self._trainer = self._create_trainer(config)

    def _train(self):
        """Calls `self.trainer.train()` and `self.trainer.validate()` once.

        You may want to override this if using a custom LR scheduler.
        """
        train_stats = self.trainer.train(max_retries=10, profile=True)
        validation_stats = self.trainer.validate(profile=True)
        stats = merge_dicts(train_stats, validation_stats)

        stats.update(
            hostplan=self.trainer.config.get("all_fluid_trial_resources", {}),
        )
        return stats

    def _save(self, checkpoint_dir):
        """Returns a path containing the trainer state."""
        checkpoint_path = os.path.join(checkpoint_dir, "trainer.checkpoint")
        self.trainer.save(checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        """Restores the trainer state.

        Override this if you have state external to the Trainer object.
        """
        return self.trainer.load(checkpoint_path)

    def _stop(self):
        """Shuts down the trainer."""
        self.trainer.shutdown()

    def _create_trainer(self, config) -> TorchTrainer:
        raise NotImplementedError

    @property
    def trainer(self) -> TorchTrainer:
        """An instantiated TorchTrainer object.

        Use this when specifying custom training procedures for Tune.
        """
        return self._trainer
