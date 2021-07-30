#!/usr/bin/env python
# coding: utf-8
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensednamer the Elastic License;
# you may not use this file except in compliance with the Elastic License.
# # Sacred experiment tracking
#
# 1. Train model (M1) on the complete dataset (D1).
# 2. Split the complete dataset (D1) into the base dataset (D2) and the update dataset (D3).
# 3. Train a new model (M2) on D2 and update it using D3.
# 4. Compare M1 and M2
#     1. Evaluation M1 and M2 on the complete dataset D1.
#     2. TODO: Compare feature importance vectors for individual data points from M1 and M2 (should be very similar)
#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error
from pathlib2 import Path

from incremental_learning.job import train, update, evaluate
from incremental_learning.config import datasets_dir, root_dir
from incremental_learning.elasticsearch import push2es

from sacred import Experiment
from sacred.observers import FileStorageObserver

experiment_name = 'sacred-test-2'
experiment_data_path = Path('/tmp/'+experiment_name)
ex = Experiment(experiment_name)
ex.observers.append(FileStorageObserver(experiment_data_path))

@ex.config
def my_config():
    dataset_name = 'ccpp'
    dataset_size = 300


def compute_metrics(_run, ytrue, m1pred, m2pred):
    m1_mae = mean_absolute_error(ytrue, m1pred)
    m1_mse = mean_squared_error(ytrue, m1pred)
    m2_mae = mean_absolute_error(ytrue, m2pred)
    m2_mse = mean_squared_error(ytrue, m2pred)
    # _run.log_scalar("after_train.mae", m1_mae)
    # _run.log_scalar("after_train.mse", m1_mse)
    # _run.log_scalar("after_update.mae", m2_mae)
    # _run.log_scalar("after_update.mse", m2_mse)
    return {"after_train.mae": m1_mae,
            "after_train.mse": m1_mse,
            "after_update.mae": m2_mae,
            "after_update.mse": m2_mse}


@ex.main
def my_main(_run, dataset_name, dataset_size):
    results = {}
    D1 = pd.read_csv(datasets_dir / '{}.csv'.format(dataset_name))
    D1.drop_duplicates(inplace=True)
    D1 = D1.sample(dataset_size)
    _run.run_logger.info("Baseline training started")
    job1 = train(dataset_name, D1, verbose=False, run=_run)
    job1.wait_to_complete(clean=False)
    results['baseline'] = {}
    results['baseline']['config'] = job1.get_config()
    results['baseline']['hyperparameters'] = job1.get_hyperparameters()
    job1.clean()
    _run.run_logger.info("Baseline training completed")
    D2, D3 = train_test_split(D1, test_size=0.2)

    _run.run_logger.info("Initial training started")
    job2 = train(dataset_name, D2, verbose=False, run=_run)
    job2.wait_to_complete(clean=False)
    results['trained_model'] = {}
    results['trained_model']['config'] = job2.get_config()
    results['trained_model']['hyperparameters'] = job2.get_hyperparameters()
    job2.clean()
    _run.run_logger.info("Initial training completed")

    _run.run_logger.info("Update started")
    job3 = update(dataset_name, D3, job2, verbose=False, run=_run)
    job3.wait_to_complete(clean=False)
    results['updated_model'] = {}
    results['updated_model']['config'] = job3.get_config()
    results['updated_model']['hyperparameters'] = job3.get_hyperparameters()
    job3.clean()
    _run.run_logger.info("Update completed")

    y_true = D1[job1.dependent_variable]

    results['baseline']['mse'] = mean_squared_error(y_true, job1.get_predictions())
    
    job2_eval = evaluate(dataset_name, D1, job2, verbose=False)
    job2_eval.wait_to_complete()
    results['trained_model']['mse'] = mean_squared_error(y_true, job2_eval.get_predictions())
    
    job3_eval = evaluate(dataset_name, D1, job3, verbose=False)
    job3_eval.wait_to_complete()
    results['updated_model']['mse'] = mean_squared_error(y_true, job3_eval.get_predictions())

    return results

if __name__ == '__main__':
    ex.run_commandline()
    push2es(data_path=experiment_data_path, name=experiment_name)

    import shutil
    shutil.rmtree(experiment_data_path)