from __future__ import annotations

import argparse
from pathlib import Path

import ray
from ray.tune import Stopper

import numpy as np
import random
import torch
import logging


logger = logging.getLogger(__name__)


def init_ray():
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("-r", "--ray-address", default="auto")
    grp.add_argument("-l", "--local", action="store_true")
    grp.add_argument("-w", "--exp", default=1,  type=int)
    grp.add_argument("-s", "--seed", default=12345,  type=int)
    args = parser.parse_args()
    ray.init(address=None if args.local else args.ray_address)

    # fix ray actors are not cleanup and takes up GPU memory
    ray.tune.ray_trial_executor.TRIAL_CLEANUP_THRESHOLD = 1

    # fix random seed
    np.random.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
    )
    logger.info("Init ray finished")
    for name in logging.root.manager.loggerDict:
        l = logging.getLogger(name)
        if name.startswith('fluid') or name.startswith('workloads'):
            l.setLevel(logging.DEBUG)
        else:
            l.setLevel(logging.INFO)
        logger.info(f"{name} at {logging.getLevelName(l.getEffectiveLevel())}")

    return args.exp, args.seed


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def detect_paths():
    tries = [
        '/nfs',
        '/tmp',
    ]
    for base in tries:
        base = Path(base)
        if base.is_dir():
            return str(base / 'data'), str(base / 'ray_results')
    raise ValueError("Can't find a suitable location to store data")


def detect_baseline_resource():
    num_gpu = ray.cluster_resources().get('GPU', 0)
    if num_gpu > 0:
        return {'cpu': 1, 'gpu': 1}
    else:
        return {'cpu': 1}


def run_options(rootfilepath):
    _, results_path = detect_paths()
    sync_to_driver = not results_path.startswith('/nfs')
    name = remove_prefix(Path(rootfilepath).stem, 'tune_')
    return {
        'name': name,
        'local_dir': results_path,
        'num_samples': int(1e10),
        'sync_to_driver': sync_to_driver,
    }


def create_cap_explore_fn(mutations, spec):
    def cap_explore(new_config):
        for name, low, high in spec:
            if name not in new_config:
                continue
            if new_config[name] < low or new_config[name] > high:
                new_config[name] = mutations[name]()
        return new_config

    return cap_explore


class MetricStopper(Stopper):
    def __init__(self, target, metric, mode):
        if mode not in ['min', 'max']:
            raise ValueError('Invalid mode')
        self.op = {
            'min': lambda a, b: np.nanmin([a, b]),
            'max': lambda a, b: np.nanmax([a, b]),
        }[mode]
        self.metric = metric
        self.target = target
        

        self.current_best = None

    def __call__(self, trial_id, result):
        if self.current_best is None:
            logger.info("Stopper: current best: None -> %f", result[self.metric])
            self.current_best = result[self.metric]
        
        if self.current_best is not None:
            old = self.current_best
            self.current_best = self.op(self.current_best, result[self.metric])
            logger.info("Stopper: current best: %f -> %f", old, self.current_best)

        return self.stop_all()

    def stop_all(self):
        if self.current_best is None:
            logger.info("Stopper: no current best, continue")
            return False

        new_best = self.op(self.target, self.current_best)
        # if current_best is better than target, stop all
        toStop = new_best == self.current_best

        if toStop:
            logger.info("Stopper: current best %f, reaches target %f, stop", self.current_best, self.target)
        else:
            logger.info("Stopper: current best %f, does not reach target %f, continue", self.current_best, self.target)

        return toStop
