#!/usr/bin/env python
# coding: utf-8

# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

#
# # Data summarization evaluation
#
# 1. From the dataset D create train dataset D1 (90%) and test dataset D0 (10%)
# 2. Train model (M1) on the dataset (D1)
#
# 3. Generate a summarization dataset from D1 and M1 using some technique (D2i) with i: 25%, 50%, 75% of data
#
# 4. Train a new model (M2i) on D2i. Identify a new set of best hyperparameters
#
# 5. Compare M1 and M2i
#     1. Compare errors of M1 and M2i on the complete dataset D1.
#     2. Compare errors M1 and M2i on D1\D2i
#     3. Compare errors on the test dataset (D0)
#     4. (Compare feature importance vectors for individual data points from M1 and M2i)

import json
import string

import pandas as pd
import numpy as np
import diversipy

from IPython import display
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from graphviz import Digraph

from ..utils.misc import train, summarize, evaluate

# ['diversity', 'random', 'kdtree', 'cover-tree', 'tree-guided', 'stratified-random']
sampling_method = 'stratified-random'

datasets = ['ccpp', 'electrical-grid-stability',
            'regression_2175_kin8nm',
            'regression_2202_elevators',
            'regression_house_16h',
            'regression_wind',
            'wine_quality_white']

for dataset_name in datasets:
    D = pd.read_csv('datasets/{}.csv'.format(dataset_name))
    D.drop_duplicates(inplace=True)

    # ## 1. From the dataset D create train dataset D1 (90%) and test dataset D0 (10%)

    D1, D0 = train_test_split(D, test_size=0.1)

    # ## 2. Train model (M1) on the dataset (D1)

    job1 = train(dataset_name, D1, verbose=False)
    job1.wait_to_complete()

    # ## 3. Generate a summarization dataset from D1 and M1 using some technique (D2i) with i: 25%, 50%, 75% of data

    D2 = {}
    for i in [0.25, 0.5, 0.75]:
        D2[i] = summarize(dataset_name=dataset_name, dataset=D1,
                          size=i, model_definition=job1.get_model_definition(), 
                          method=sampling_method, verbose=False, 
                          dependent_variable=job1.dependent_variable)

    # ## 4. Train a new model (M2i) on D2i

    jobs2 = {}
    for fraction, D2i in D2.items():
        job = train(dataset_name, D2i, verbose=False)
        job.wait_to_complete()
        jobs2[fraction] = job

    # ## 5. Compare M1 and M2i

    # ### A. Compare errors of M1 and M2i on the complete dataset D1.

    y_true = D1[job1.dependent_variable]

    y_M1 = job1.get_predictions()
    y_M2 = {}
    for i, job2i in jobs2.items():
        eval_job = evaluate(dataset_name=dataset_name,
                            dataset=D1, model=job2i.model, verbose=False)
        success = eval_job.wait_to_complete()
        if not success:
            print('{} failed'.format(i))
            break
        y_M2[i] = eval_job.get_predictions()
    mse1 = [mean_squared_error(y_true, y_M1)] + \
        [mean_squared_error(y_true, y) for y in y_M2.values()]

    # ### B. Compare errors M1 and M2i on D1\D2i

    ds1 = set([tuple(line) for line in D1.values])
    D1notD2 = {}
    for key, D2i in D2.items():
        ds2 = set([tuple(line) for line in D2i.values])
        D1notD2[key] = pd.DataFrame(
            list(ds1.difference(ds2)), columns=D1.columns)

    y_M1 = {}
    y_M2 = {}
    for key, job2i in jobs2.items():
        eval_job = evaluate(dataset_name=dataset_name,
                            dataset=D1notD2[key], model=job1.model, verbose=False)
        success = eval_job.wait_to_complete()
        if not success:
            print('M1 evaluation for {} failed'.format(key))
            break
        y_M1[key] = eval_job.get_predictions()

        eval_job = evaluate(dataset_name=dataset_name,
                            dataset=D1notD2[key], model=job2i.model, verbose=False)
        success = eval_job.wait_to_complete()
        if not success:
            print('M2 evaluation for {} failed'.format(key))
            break
        y_M2[key] = eval_job.get_predictions()

    mse2 = [mean_squared_error(D1notD2[k][job1.dependent_variable], y) for k, y in y_M1.items(
    )] + [mean_squared_error(D1notD2[k][job1.dependent_variable], y) for k, y in y_M2.items()]

    # ### C. Compare errors on the test dataset (D0)
    y_true = D0[job1.dependent_variable]

    eval_job = evaluate(dataset_name=dataset_name,
                        dataset=D0, model=job1.model, verbose=False)
    success = eval_job.wait_to_complete()
    y_M1 = eval_job.get_predictions()

    y_M2 = {}
    for i, job2i in jobs2.items():
        eval_job = evaluate(dataset_name=dataset_name,
                            dataset=D0, model=job2i.model, verbose=False)
        success = eval_job.wait_to_complete()
        if not success:
            print('{} failed'.format(i))
            break
        y_M2[i] = eval_job.get_predictions()

    mse3 = [mean_squared_error(y_true, y_M1)] + \
        [mean_squared_error(y_true, y) for y in y_M2.values()]

    print("Summary for {} with {}".format(dataset_name, sampling_method))
    print(mse1+mse2+mse3)
