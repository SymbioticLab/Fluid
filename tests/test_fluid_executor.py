# coding: utf-8
import json
import logging
import unittest

import ray
from ray.cluster_utils import Cluster
from ray.rllib import _register_all

# from ray.rllib import _register_all
from ray.tune import Trainable
from ray.tune.registry import TRAINABLE_CLASS, _global_registry
from ray.tune.resources import Resources
from ray.tune.result import TRAINING_ITERATION
from ray.tune.suggest import BasicVariantGenerator
from ray.tune.trial import Checkpoint, Trial

from fluid.fluid_executor import FluidExecutor


class FluidExecutorTest(unittest.TestCase):
    def setUp(self):
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        logger = logging.getLogger("ray.tune.registry")
        logger.setLevel(logging.INFO)
        logger = logging.getLogger("fluid.fluid_executor")
        logger.setLevel(logging.DEBUG)

        # register the __fake trainable
        _register_all()
        ray.init()
        self.trial_executor = FluidExecutor()

    def tearDown(self):
        ray.shutdown()
        _register_all()

    def testStartStop(self):
        trial = Trial("__fake")
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)

        running = self.trial_executor.get_running_trials()
        self.assertEqual(1, len(running))
        self.trial_executor.stop_trial(trial)

    def testAsyncSave(self):
        """Tests that saved checkpoint value not immediately set."""
        trial = Trial("__fake")
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        self.assertEqual(Trial.RUNNING, trial.status)
        trial.last_result = self.trial_executor.fetch_result(trial)
        checkpoint = self.trial_executor.save(trial, Checkpoint.PERSISTENT)
        self.assertEqual(checkpoint, trial.saving_to)
        self.assertEqual(trial.checkpoint.value, None)
        self.process_trial_save(trial)
        self.assertEqual(checkpoint, trial.checkpoint)
        self.trial_executor.stop_trial(trial)
        self.assertEqual(Trial.TERMINATED, trial.status)

    def testSaveRestore(self):
        trial = Trial("__fake")
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        self.assertEqual(Trial.RUNNING, trial.status)
        trial.last_result = self.trial_executor.fetch_result(trial)
        self.trial_executor.save(trial, Checkpoint.PERSISTENT)
        self.process_trial_save(trial)
        self.trial_executor.restore(trial)
        self.trial_executor.stop_trial(trial)
        self.assertEqual(Trial.TERMINATED, trial.status)

    def testPauseResume(self):
        """Tests that pausing works for trials in flight."""
        trial = Trial("__fake")
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        self.assertEqual(Trial.RUNNING, trial.status)
        self.trial_executor.pause_trial(trial)
        self.assertEqual(Trial.PAUSED, trial.status)
        self.trial_executor.start_trial(trial)
        self.assertEqual(Trial.RUNNING, trial.status)
        self.trial_executor.stop_trial(trial)
        self.assertEqual(Trial.TERMINATED, trial.status)

    def testSavePauseResumeErrorRestore(self):
        """Tests that pause checkpoint does not replace restore checkpoint."""
        trial = Trial("__fake")
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        trial.last_result = self.trial_executor.fetch_result(trial)
        # Save
        checkpoint = self.trial_executor.save(trial, Checkpoint.PERSISTENT)
        self.assertEqual(Trial.RUNNING, trial.status)
        self.assertEqual(checkpoint.storage, Checkpoint.PERSISTENT)
        # Process save result (simulates trial runner)
        self.process_trial_save(trial)
        # Train
        self.trial_executor.continue_training(trial)
        trial.last_result = self.trial_executor.fetch_result(trial)
        # Pause
        self.trial_executor.pause_trial(trial)
        self.assertEqual(Trial.PAUSED, trial.status)
        self.assertEqual(trial.checkpoint.storage, Checkpoint.MEMORY)
        # Resume
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        self.assertEqual(Trial.RUNNING, trial.status)
        # Error
        trial.set_status(Trial.ERROR)
        # Restore
        self.trial_executor.restore(trial)
        self.trial_executor.stop_trial(trial)
        self.assertEqual(Trial.TERMINATED, trial.status)

    def testStartFailure(self):
        _global_registry.register(TRAINABLE_CLASS, "asdf", None)
        trial = Trial("asdf", resources=Resources(1, 0))
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        self.assertEqual(Trial.ERROR, trial.status)

    def testPauseResume2(self):
        """Tests that pausing works for trials being processed."""
        trial = Trial("__fake")
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        self.assertEqual(Trial.RUNNING, trial.status)
        self.trial_executor.fetch_result(trial)
        checkpoint = self.trial_executor.pause_trial(trial)
        self.assertEqual(Trial.PAUSED, trial.status)
        self.trial_executor.start_trial(trial, checkpoint)
        self.trial_executor.on_no_available_trials(None)
        self.assertEqual(Trial.RUNNING, trial.status)
        self.trial_executor.stop_trial(trial)
        self.assertEqual(Trial.TERMINATED, trial.status)

    def testPauseUnpause(self):
        """Tests that unpausing works for trials being processed."""
        trial = Trial("__fake")
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        self.assertEqual(Trial.RUNNING, trial.status)

        trial.last_result = self.trial_executor.fetch_result(trial)
        self.assertEqual(trial.last_result.get(TRAINING_ITERATION), 1)

        print("Pausing trial")
        self.trial_executor.pause_trial(trial)
        self.assertEqual(Trial.PAUSED, trial.status)

        print("Unpausing trial")
        self.trial_executor.unpause_trial(trial)
        self.assertEqual(Trial.PENDING, trial.status)

        print("Start trial again")
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        self.assertEqual(Trial.RUNNING, trial.status)

        print("fetch result")
        trial.last_result = self.trial_executor.fetch_result(trial)
        self.assertEqual(trial.last_result.get(TRAINING_ITERATION), 2)

        print("stop trial")
        self.trial_executor.stop_trial(trial)
        self.assertEqual(Trial.TERMINATED, trial.status)

    def testNoResetTrial(self):
        """Tests that reset handles NotImplemented properly."""
        trial = Trial("__fake")
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        exists = self.trial_executor.reset_trial(trial, {}, "modified_mock")
        self.assertEqual(exists, False)
        self.assertEqual(Trial.RUNNING, trial.status)

    def testResetTrial(self):
        """Tests that reset works as expected."""

        class B(Trainable):
            def _train(self):
                return dict(timesteps_this_iter=1, done=True)

            def reset_config(self, config):
                self.config = config
                return True

        trials = self.generate_trials(
            {
                "run": B,
                "config": {"foo": 0},
            },
            "grid_search",
        )
        trial = trials[0]
        self.trial_executor.start_trial(trial)
        self.trial_executor.on_no_available_trials(None)
        exists = self.trial_executor.reset_trial(trial, {"hi": 1}, "modified_mock")
        self.assertEqual(exists, True)
        self.assertEqual(trial.config.get("hi"), 1)
        self.assertEqual(trial.experiment_tag, "modified_mock")
        self.assertEqual(Trial.RUNNING, trial.status)

    @staticmethod
    def generate_trials(spec, name):
        suggester = BasicVariantGenerator()
        suggester.add_configurations({name: spec})
        return suggester.next_trials()

    def process_trial_save(self, trial):
        """Simulates trial runner save."""
        checkpoint = trial.saving_to
        checkpoint_value = self.trial_executor.fetch_result(trial)
        checkpoint.value = checkpoint_value
        trial.on_checkpoint(checkpoint)


class FluidExecutorQueueTest(unittest.TestCase):
    def setUp(self):
        self.cluster = Cluster(
            initialize_head=True,
            connect=True,
            head_node_args={
                "num_cpus": 1,
                "_internal_config": json.dumps({"num_heartbeats_timeout": 10}),
            },
        )
        self.trial_executor = FluidExecutor()
        # Pytest doesn't play nicely with imports
        _register_all()

    def tearDown(self):
        ray.shutdown()
        self.cluster.shutdown()
        _register_all()  # re-register the evicted objects

    def testQueueTrial(self):
        """Tests that reset handles NotImplemented properly."""

        def create_trial(cpu, gpu=0):
            return Trial("__fake", resources=Resources(cpu=cpu, gpu=gpu))

        cpu_only = create_trial(1, 0)
        self.assertTrue(self.trial_executor.has_resources(cpu_only.resources))
        self.trial_executor.start_trial(cpu_only)
        self.trial_executor.on_no_available_trials(None)

        gpu_only = create_trial(0, 1)
        self.assertTrue(self.trial_executor.has_resources(gpu_only.resources))
