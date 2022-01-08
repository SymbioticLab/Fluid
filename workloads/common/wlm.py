import ConfigSpace as CS
import numpy as np
import torch
import torch.nn as nn
from ray import tune
from ray.util.sgd.torch.training_operator import TrainingOperator
from ray.util.sgd.utils import BATCH_SIZE
from torch.utils.data import DataLoader, TensorDataset

import workloads.common as com
from workloads.word_language_model import data as wlm_data
from workloads.word_language_model import main as wlm_main
from workloads.word_language_model import model as wlm_model


def exp_metric():
    return dict(metric="val_ppl", mode="min")


def create_stopper():
    return com.MetricStopper(140, **exp_metric())


def data_creator(config):
    corpus = wlm_data.Corpus()
    train_loader = DataLoader(
        TensorDataset(corpus.train), batch_size=config[BATCH_SIZE], shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(corpus.valid), batch_size=config[BATCH_SIZE], shuffle=False
    )

    return train_loader, val_loader


def model_creator(config):
    corpus = wlm_data.Corpus()
    ntokens = len(corpus.dictionary)
    """
    if config.get('model') == 'Transformer':
        m = wlm_model.TransformerModel(
            ntokens,
            config.get('half_emsize') * 2 * config.get('nhead'),
            config.get('nhead'),
            config.get('nhid'),
            config.get('nlayers'),
            config.get('dropout')
        )
    else:
        m = wlm_model.RNNModel(
            config.get('model'),
            ntokens,
            config.get('half_emsize') ,
            config.get('nhid'),
            config.get('nlayers'),
            config.get('dropout'),
            False
        )
    """
    m = wlm_model.RNNModel(
        config.get("model"),
        ntokens,
        config.get("half_emsize") * 2 * config.get("nhead"),
        config.get("nhid"),
        config.get("nlayers"),
        config.get("dropout"),
        False,
    )
    return m


def loss_creator(config):
    return nn.NLLLoss()


def optimizer_creator(model, config):
    return []


class WLMOperator(TrainingOperator):
    def setup(self, config):
        corpus = wlm_data.Corpus()
        self.train_data = wlm_main.batchify(
            corpus.train, config[BATCH_SIZE], torch.device("cpu")
        )
        self.val_data = wlm_main.batchify(
            corpus.valid, config[BATCH_SIZE], torch.device("cpu")
        )

        if torch.cuda.is_available():
            self.train_data = self.train_data.cuda(non_blocking=True)
            self.val_data = self.val_data.cuda(non_blocking=True)

        self.ntokens = len(corpus.dictionary)
        self.clip = config["clip"]
        self.lr = config["lr"]
        self.bptt = config["bptt"]
        self.best_val_loss = None

    def validate(self, val_iterator, info):
        val_loss = wlm_main.evaluate(
            self.val_data,
            self.model,
            self.criterion,
            self.ntokens,
            self.config["model"],
            self.config[BATCH_SIZE],
            self.bptt,
        )

        if not self.best_val_loss or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            self.lr /= 4.0

        return {
            "val_loss": val_loss,
            "val_ppl": np.exp(val_loss),
        }

    def train_epoch(self, iterator, info):
        loss = wlm_main.train(
            None,
            self.model,
            self.criterion,
            self.ntokens,
            self.train_data,
            self.config[BATCH_SIZE],
            self.bptt,
            self.config["model"],
            self.lr,
            self.clip,
            log_interval=None,
        )
        return {
            "train_loss": loss,
            "train_ppl": np.exp(loss),
        }


def create_sample_space():
    mutations = {
        BATCH_SIZE: lambda *_: np.random.randint(10, 31),
        "bptt": lambda *_: np.random.randint(30, 41),
        "clip": lambda *_: np.random.uniform(0.1, 0.5),
        "model": ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"],
        # 'model': ["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"],
        # half_emsize is multiplied by 2 * nhead to get emsize, and the conceptual range is 100, 300
        "half_emsize": lambda *_: np.random.randint(25, 76),
        "nhead": lambda *_: np.random.randint(1, 3),
        "nhid": lambda *_: np.random.randint(100, 301),
        "nlayers": lambda *_: np.random.randint(2, 5),
        "dropout": lambda *_: np.random.uniform(0.1, 0.5),
        "lr": lambda *_: np.random.uniform(10, 30),
    }

    cap_explore = com.create_cap_explore_fn(
        mutations,
        [
            (BATCH_SIZE, 10, 31),
            ("bptt", 30, 41),
            ("clip", 0.1, 0.5),
            # half_emsize is multiplied by 2 * nhead to get emsize, and the conceptual range is 100, 300
            ("half_emsize", 25, 76),
            ("nhead", 1, 3),
            ("nhid", 100, 301),
            ("nlayers", 2, 5),
            ("dropout", 0.1, 0.5),
            ("lr", 10, 30),
        ],
    )

    return mutations, cap_explore


def create_search_space():
    def _create_tune_sample_fn(val):
        if isinstance(val, list):
            return tune.choice(val)
        else:
            return tune.sample_from(val)

    return {
        key: _create_tune_sample_fn(val)
        for key, val in create_sample_space()[0].items()
    }


def create_ch():
    # BOHB uses ConfigSpace for their hyperparameter search space
    config_space = CS.ConfigurationSpace()
    # batch_size
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(name=BATCH_SIZE, lower=10, upper=30)
    )
    # bptt
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(name="bptt", lower=30, upper=40)
    )
    # clip
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(name="clip", lower=0.1, upper=0.5)
    )
    # model ['', '', '']
    cs_model = CS.CategoricalHyperparameter(
        name="model",
        # choices=["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"]
        choices=["RNN_TANH", "RNN_RELU", "LSTM", "GRU"],
    )
    config_space.add_hyperparameter(cs_model)
    # emsize
    # emsize need to be even, and divisible by nhead
    # half_emsize is multiplied by 2 * nhead to get emsize, and the conceptual range is 100, 300
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(name="half_emsize", lower=25, upper=75)
    )
    # nhead
    cs_nhead = CS.UniformIntegerHyperparameter(name="nhead", lower=1, upper=3)
    config_space.add_hyperparameter(cs_nhead)
    # nhid
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(name="nhid", lower=150, upper=250)
    )
    # nlayers
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(name="nlayers", lower=2, upper=4)
    )
    # dropout
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(name="dropout", lower=0.1, upper=0.5)
    )
    # lr
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(name="lr", lower=10, upper=30)
    )

    return config_space


def create_skopt_space():
    return [
        (10, 30),
        (30, 40),
        (0.1, 0.5),
        # ["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"],
        ["RNN_TANH", "RNN_RELU", "LSTM", "GRU"],
        (50, 150),
        (1, 3),
        (150, 250),
        (2, 4),
        (0.1, 0.5),
        (10, 30),
    ], [
        BATCH_SIZE,
        "bptt",
        "clip",
        "model",
        "half_emsize",
        "nhead",
        "nhid",
        "nlayers",
        "dropout",
        "lr",
    ]
