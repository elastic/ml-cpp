#!/usr/bin/env python
# coding: utf-8
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
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

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error

from incremental_learning.job import train, update, evaluate
from incremental_learning.config import datasets_dir, root_dir

from sacred import Experiment
ex = Experiment('my_experiment')

# In[2]:


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


@ex.automain
def my_main(_run, dataset_name, dataset_size):
    D1 = pd.read_csv(datasets_dir / '{}.csv'.format(dataset_name))
    D1.drop_duplicates(inplace=True)
    D1 = D1.sample(dataset_size)
    _run.run_logger.info("Baseline training started")
    job1 = train(dataset_name, D1, verbose=False, run=_run)
    job1.wait_to_complete()
    _run.run_logger.info("Baseline training completed")
    D2, D3 = train_test_split(D1, test_size=0.2)

    _run.run_logger.info("Initial training started")
    job2 = train(dataset_name, D2, verbose=False, run=_run)
    job2.wait_to_complete()
    _run.run_logger.info("Initial training completed")

    _run.run_logger.info("Update started")
    job3 = update(dataset_name, D3, job2, verbose=False, run=_run)
    job3.wait_to_complete()
    _run.run_logger.info("Update completed")

    y_true = D1[job1.dependent_variable]
    y_M1 = job1.get_predictions()
    eval_job = evaluate(dataset_name, D1, job2, verbose=False)
    success = eval_job.wait_to_complete()
    if not success:
        print('Evaluation failed')
    y_M2 = eval_job.get_predictions()

    return compute_metrics(_run, y_true, y_M1, y_M2)
