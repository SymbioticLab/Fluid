#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:10:09 2020

@author: liujiachen
"""

from pathlib import Path
 
from ray.util.sgd.utils import BATCH_SIZE
from ray import tune

import workloads.common as com
from workloads.common import mnist_upgrade as workload
import torch 
import torch.nn as nn

DATA_PATH, RESULTS_PATH = com.detect_paths()
EXP_NAME = com.remove_prefix(Path(__file__).stem, 'tune_')

EPOCH_SIZE = 512
TEST_SIZE = 256

 
def create_grid_space():
    # median variance
    # number of sample : 81
    mutations = {
        # distribution for resampling
        "lr": [0.0001,0.001,0.00001],
        "momentum": [0.1],
        'hidden_layer1': [128, 256, 512],
        'hidden_layer2':  [32, 64, 128],
        "drop_out": [0.1, 0.2,0.5],
    }
    return mutations


def create_grid_search_space():
    mutations = create_grid_space()
    return {
        key: tune.sample_from(val)
        for key, val in mutations.items()
    }

def train(model, optimizer, train_loader,config, device=None):
    device = device or torch.device("cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss =  nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def train_mnist(config):
    #track.init()
    train_loader, test_loader = workload.data_creator(config)
    model = workload.model_creator(config)
    optimizer = workload.optimizer_creator(model, config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    acc = 0
    for i in range(81):
        train(model, optimizer, train_loader,config, device)
        acc = test(model, test_loader, device)
        if i % 10 == 0 :
            print("Train epoch ", i, " | Test accuaracy: " , acc)
        # track.log(mean_accuracy=acc)
        

def setup_tune_scheduler():
    search_space =  create_grid_search_space()
    return dict(
        config=search_space, 
    )
    
def main():
    _, sd = com.init_ray()
    analysis = tune.run(
        train_mnist, **setup_tune_scheduler())

    dfs = analysis.trial_dataframes
    for logdir, df in dfs.items():
        ld = Path(logdir)
        df.to_csv(ld / 'trail_dataframe.csv')



if __name__ == '__main__':
    main()
