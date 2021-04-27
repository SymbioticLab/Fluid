#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:29:15 2020

@author: liujiachen
"""
from ray.util.sgd.utils import BATCH_SIZE


def create_grid_space_1():
    # number of sample : 81
    mutations = {
        # distribution for resampling
        "lr": [0.001, 0.01, 0.1],
        # allow perturbations within this set of categorical values
        "momentum": [0.9],
        "conv_1": [64, 128, 256],
        "conv_2": [64, 128, 256],
        "dropout_1": [0.01, 0.05, 0.1],
        "dropout_2": [0.1],
        "dense_1": [1024],
    }
    return mutations


def create_grid_space_9():
    # number of sample : 27
    mutations = {
        # distribution for resampling
        "lr": [0.01],
        # allow perturbations within this set of categorical values
        "momentum": [0.9],
        "conv_1": [64],
        "conv_2": [64, 128, 256],
        "dropout_1": [0.01, 0.05, 0.1],
        "dropout_2": [0.1],
        "dense_1": [128, 512, 1024],
    }
    return mutations


def create_grid_space_10():
    # number of sample : 9
    mutations = {
        # distribution for resampling
        "lr": [0.01],
        # allow perturbations within this set of categorical values
        "momentum": [0.9],
        "conv_1": [64],
        "conv_2": [64, 128, 256],
        "dropout_1": [0.05],
        "dropout_2": [0.1],
        "dense_1": [128, 512, 1024],
    }
    return mutations


def create_grid_space_11():
    # number of sample : 3
    mutations = {
        # distribution for resampling
        "lr": [0.01],
        # allow perturbations within this set of categorical values
        "momentum": [0.9],
        "conv_1": [64],
        "conv_2": [128],
        "dropout_1": [0.05],
        "dropout_2": [0.1],
        "dense_1": [128, 512, 1024],
    }
    return mutations


def create_grid_space_12():
    # no variance
    # number of sample : 3
    mutations = {
        # distribution for resampling
        "lr": [0.01, 0.001, 0.1],
        # allow perturbations within this set of categorical values
        "momentum": [0.9],
        "conv_1": [64],
        "conv_2": [128],
        "dropout_1": [0.05],
        "dropout_2": [0.1],
        "dense_1": [1024],
    }
    return mutations


def create_grid_space_14():
    # no variance
    # number of sample : 3
    mutations = {
        # distribution for resampling
        "lr": [0.01],
        # allow perturbations within this set of categorical values
        "momentum": [0.9],
        "conv_1": [64],
        "conv_2": [32, 128, 512],
        "dropout_1": [0.05],
        "dropout_2": [0.1],
        "dense_1": [1024],
    }
    return mutations


def create_grid_space_24():
    # no variance
    # number of sample : 5
    mutations = {
        # distribution for resampling
        "lr": [0.01],
        # allow perturbations within this set of categorical values
        "momentum": [0.9],
        "conv_1": [64],
        "conv_2": [32, 64, 128, 256, 512],
        "dropout_1": [0.05],
        "dropout_2": [0.1],
        "dense_1": [1024],
    }
    return mutations


def create_grid_space_25():
    # no variance
    # number of sample : 5
    mutations = {
        # distribution for resampling
        "lr": [0.01],
        # allow perturbations within this set of categorical values
        "momentum": [0.9],
        "conv_1": [64],
        "conv_2": [16, 32, 64, 128, 256, 512, 1024],
        "dropout_1": [0.05],
        "dropout_2": [0.1],
        "dense_1": [1024],
    }
    return mutations


def create_grid_wlm_1():
    # number of sample : 81
    mutations = {
        BATCH_SIZE: [10, 20, 30],
        "bptt": [30],
        "clip": [0.1, 0.3, 0.5],
        # 'model': ["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"],
        "model": ["RNN_TANH", "LSTM", "GRU"],
        "half_emsize": [100],
        "nhead": [2],
        "nhid": [200],
        "nlayers": [2, 3, 4],
        "dropout": [0.1],
        "lr": [10],
    }

    return mutations


def create_grid_wlm_9():
    # number of sample : 27
    mutations = {
        BATCH_SIZE: [10, 20, 30],
        "bptt": [30],
        "clip": [0.13],
        # 'model': ["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"],
        "model": ["RNN_TANH", "LSTM", "GRU"],
        "half_emsize": [100],
        "nhead": [2],
        "nhid": [200],
        "nlayers": [2, 3, 4],
        "dropout": [0.1],
        "lr": [10],
    }

    return mutations


def create_grid_wlm_10():
    # number of sample : 9
    mutations = {
        BATCH_SIZE: [30],
        "bptt": [30],
        "clip": [0.13],
        # 'model': ["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"],
        "model": ["RNN_TANH", "LSTM", "GRU"],
        "half_emsize": [100],
        "nhead": [2],
        "nhid": [200],
        "nlayers": [2, 3, 4],
        "dropout": [0.1],
        "lr": [10],
    }

    return mutations


def create_grid_wlm_11():
    # number of sample : 3
    mutations = {
        BATCH_SIZE: [30],
        "bptt": [30],
        "clip": [0.13],
        # 'model': ["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"],
        "model": ["LSTM"],
        "half_emsize": [100],
        "nhead": [2],
        "nhid": [200],
        "nlayers": [2, 3, 4],
        "dropout": [0.1],
        "lr": [10],
    }

    return mutations


def create_grid_wlm_12():
    # number of sample : 3
    # no variance
    mutations = {
        BATCH_SIZE: [30],
        "bptt": [30],
        "clip": [0.13],
        # 'model': ["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"],
        "model": ["LSTM"],
        "half_emsize": [100],
        "nhead": [2],
        "nhid": [200],
        "nlayers": [4],
        "dropout": [0.1],
        "lr": [10, 15, 20],
    }

    return mutations


def create_grid_wlm_14():
    # number of sample : 3
    mutations = {
        BATCH_SIZE: [30],
        "bptt": [30],
        "clip": [0.13],
        # 'model': ["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"],
        "model": ["RNN_TANH", "LSTM", "GRU"],
        "half_emsize": [100],
        "nhead": [2],
        "nhid": [200],
        "nlayers": [4],
        "dropout": [0.1],
        "lr": [10],
    }

    return mutations


def create_grid_wlm_24():
    # number of sample : 5
    mutations = {
        BATCH_SIZE: [30],
        "bptt": [30],
        "clip": [0.1, 0.2, 0.3, 0.4, 0.5],
        # 'model': ["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"],
        "model": ["LSTM"],
        "half_emsize": [100],
        "nhead": [2],
        "nhid": [200],
        "nlayers": [4],
        "dropout": [0.1],
        "lr": [10],
    }

    return mutations


def create_grid_wlm_25():
    # number of sample : 7
    mutations = {
        BATCH_SIZE: [30],
        "bptt": [35],
        "clip": [0.1, 0.15, 0.2, 0.45, 0.3, 0.4, 0.5],
        # 'model': ["RNN_TANH", "RNN_RELU", "LSTM", "GRU", "Transformer"],
        "model": ["LSTM"],
        "half_emsize": [100],
        "nhead": [2],
        "nhid": [200],
        "nlayers": [4],
        "dropout": [0.1],
        "lr": [10],
    }

    return mutations


def create_grid_dcgan_1():
    # number of sample : 81
    mutations = {
        # distribution for resampling
        "lr": [0.001, 0.01, 0.1],
        # allow perturbations within this set of categorical values
        "momentum": [0.9],
        "conv_1": [64, 128, 256],
        "conv_2": [64, 128, 256],
        "dropout_1": [0.01, 0.05, 0.1],
        "dropout_2": [0.1],
        "dense_1": [1024],
    }
    return mutations


"""
def create_grid_search_space():
    mutations = create_grid_space()
    return {
        key: tune.sample_from(val)
        for key, val in mutations.items()
    }
"""
