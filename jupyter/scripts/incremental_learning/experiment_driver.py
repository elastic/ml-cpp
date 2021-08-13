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


import logging
import shutil

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from deepmerge import always_merger
from pathlib2 import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.model_selection import train_test_split

from incremental_learning.config import datasets_dir, logger, root_dir
from incremental_learning.elasticsearch import push2es
from incremental_learning.job import evaluate, train, update
from incremental_learning.storage import download_dataset
from incremental_learning.transforms import resample_metric_features

experiment_name = 'generic-train-update'
experiment_data_path = Path('/tmp/'+experiment_name)
ex = Experiment(experiment_name)
ex.observers.append(FileStorageObserver(experiment_data_path))
ex.logger = logger


@ex.config
def my_config():
    dataset_name = 'ccpp'
    test_fraction = 0.2
    threads = 1


def compute_regression_metrics(y_true, baseline_model_predictions, trained_model_predictions, updated_model_predictions):
    baseline_model_mae = metrics.mean_absolute_error(
        y_true, baseline_model_predictions)
    trained_model_mae = metrics.mean_absolute_error(
        y_true, trained_model_predictions)
    updated_model_mae = metrics.mean_absolute_error(
        y_true, updated_model_predictions)
    baseline_model_mse = metrics.mean_squared_error(
        y_true, baseline_model_predictions)
    trained_model_mse = metrics.mean_squared_error(
        y_true, trained_model_predictions)
    updated_model_mse = metrics.mean_squared_error(
        y_true, updated_model_predictions)
    scores = \
        {
            "baseline": {"mae": baseline_model_mae,
                         "mse": baseline_model_mse},
            "trained_model": {"mae": trained_model_mae,
                              "mse": trained_model_mse},
            "updated_model": {"mae": updated_model_mae,
                              "mse": updated_model_mse},
        }
    return scores


def compute_classification_metrics(y_true, baseline_model_predictions, trained_model_predictions, updated_model_predictions):
    baseline_model_acc = metrics.accuracy_score(
        y_true, baseline_model_predictions)
    trained_model_acc = metrics.accuracy_score(
        y_true, trained_model_predictions)
    updated_model_acc = metrics.accuracy_score(
        y_true, updated_model_predictions)
    baseline_model_precision = metrics.precision_score(
        y_true, baseline_model_predictions)
    trained_model_precision = metrics.precision_score(
        y_true, trained_model_predictions)
    updated_model_precision = metrics.precision_score(
        y_true, updated_model_predictions)
    baseline_model_recall = metrics.recall_score(
        y_true, baseline_model_predictions)
    trained_model_recall = metrics.recall_score(
        y_true, trained_model_predictions)
    updated_model_recall = metrics.recall_score(
        y_true, updated_model_predictions)
    baseline_model_roc_auc = metrics.roc_auc_score(
        y_true, baseline_model_predictions)
    trained_model_roc_auc = metrics.roc_auc_score(
        y_true, trained_model_predictions)
    updated_model_roc_auc = metrics.roc_auc(y_true, updated_model_predictions)
    scores = \
        {
            "baseline": {
                "acc": baseline_model_acc,
                "precision": baseline_model_precision,
                "recall": baseline_model_recall,
                "roc_auc": baseline_model_roc_auc
            },
            "trained_model": {
                "acc": trained_model_acc,
                "precision": trained_model_precision,
                "recall": trained_model_recall,
                "roc_auc": trained_model_roc_auc

            },
            "updated_model": {
                "acc": updated_model_acc,
                "precision": updated_model_precision,
                "recall": updated_model_recall,
                "roc_auc": updated_model_roc_auc
            },
        }
    return scores


@ex.capture
def read_dataset(_run, dataset_name):
    download_successful = download_dataset(dataset_name)
    if download_successful == False:
        _run.run_logger.error("Data is not available")
        exit(1)
    dataset = pd.read_csv(datasets_dir / '{}.csv'.format(dataset_name))
    dataset.drop_duplicates(inplace=True)
    return dataset


@ex.capture
def transform_dataset(dataset: pd.DataFrame, transform_name: str, transform_parameters, _seed: int) -> pd.DataFrame:
    if transform_name == 'resample_metric_features':
        transformed_dataset = resample_metric_features(data_frame=dataset,
                                                       seed=_seed, 
                                                       fraction=transform_parameters['fraction'], 
                                                       magnitude=transform_parameters['magnitude'],
                                                       features=transform_parameters['features'])
        return transformed_dataset
    # TODO extend to other transforms
    else:
        raise NotImplementedError("This dataset transform is not integrated yet.")


@ex.main
def my_main(_run, dataset_name, test_fraction):
    results = {}

    original_dataset = read_dataset()
    train_dataset, test1_dataset = train_test_split(original_dataset, test_size=test_fraction)
    update_dataset = transform_dataset(dataset=train_dataset)
    test2_dataset = transform_dataset(dataset=test1_dataset)

    baseline_dataset = pd.concat([train_dataset, update_dataset])
    test_dataset = pd.concat([test1_dataset, test2_dataset])
    
    _run.run_logger.info("Baseline training started")
    job1 = train(dataset_name, baseline_dataset, verbose=False, run=_run)
    elapsed_time = job1.wait_to_complete(clean=False)
    results['baseline'] = {}
    results['baseline']['config'] = job1.get_config()
    results['baseline']['hyperparameters'] = job1.get_hyperparameters()
    results['baseline']['elapsed_time'] = elapsed_time
    job1.clean()
    _run.run_logger.info("Baseline training completed")

    dependent_variable = job1.dependent_variable

    _run.run_logger.info("Initial training started")
    job2 = train(dataset_name, train_dataset, verbose=False, run=_run)
    elapsed_time = job2.wait_to_complete(clean=False)
    results['trained_model'] = {}
    results['trained_model']['config'] = job2.get_config()
    results['trained_model']['hyperparameters'] = job2.get_hyperparameters()
    results['trained_model']['elapsed_time'] = elapsed_time
    job2.clean()
    _run.run_logger.info("Initial training completed")

    _run.run_logger.info("Update started")
    job3 = update(dataset_name, update_dataset, job2, verbose=False, run=_run)
    elapsed_time = job3.wait_to_complete(clean=False)
    results['updated_model'] = {}
    results['updated_model']['config'] = job3.get_config()
    results['updated_model']['hyperparameters'] = job3.get_hyperparameters()
    results['updated_model']['elapsed_time'] = elapsed_time
    job3.clean()
    _run.run_logger.info("Update completed")

    y_true = test_dataset[dependent_variable]

    job1_eval = evaluate(dataset_name, test_dataset, job1, verbose=False)
    job1_eval.wait_to_complete()

    job2_eval = evaluate(dataset_name, test_dataset, job2, verbose=False)
    job2_eval.wait_to_complete()

    job3_eval = evaluate(dataset_name, test_dataset, job3, verbose=False)
    job3_eval.wait_to_complete()

    scores = {}

    if job1.is_regression():
        scores = compute_regression_metrics(y_true,
                                            job1_eval.get_predictions(),
                                            job2_eval.get_predictions(),
                                            job3_eval.get_predictions())
    elif job1.is_classification():
        scores = compute_classification_metrics(y_true,
                                                job1_eval.get_predictions(),
                                                job2_eval.get_predictions(),
                                                job3_eval.get_predictions())
    else:
        _run.run_logger.warning(
            "Job is neither regression nor classification. No metric scores are available.")
    return always_merger.merge(results, scores)


if __name__ == '__main__':
    run = ex.run_commandline()
    run_data_path = experiment_data_path / str(run._id)
    push2es(data_path=run_data_path, name=experiment_name)
    shutil.rmtree(run_data_path)
