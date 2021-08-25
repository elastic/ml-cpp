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


import shutil

import pandas as pd
import sklearn.metrics as metrics
from deepmerge import always_merger
from pathlib2 import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.model_selection import train_test_split

from incremental_learning.config import datasets_dir, logger
from incremental_learning.elasticsearch import push2es
from incremental_learning.job import evaluate, train, update
from incremental_learning.storage import download_dataset
from incremental_learning.transforms import partition_on_metric_ranges
from incremental_learning.transforms import regression_category_drift
from incremental_learning.transforms import resample_metric_features
from incremental_learning.transforms import rotate_metric_features
from incremental_learning.transforms import shift_metric_features

experiment_name = 'generic-train-update'
experiment_data_path = Path('/tmp/'+experiment_name)
ex = Experiment(experiment_name)
ex.observers.append(FileStorageObserver(experiment_data_path))
ex.logger = logger


@ex.config
def my_config():
    verbose = False
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
def transform_dataset(dataset: pd.DataFrame,
                      test_fraction: float,
                      transform_name: str,
                      transform_parameters,
                      _seed: int) -> pd.DataFrame:
    if transform_name == 'partition_on_metric_ranges':
        train_dataset, update_dataset = partition_on_metric_ranges(
            dataset=dataset,
            seed=_seed,
            metric_features=transform_parameters['metric_features'])
        train_dataset, test1_dataset = train_test_split(train_dataset, test_size=test_fraction)
        update_dataset, test2_dataset = train_test_split(update_dataset, test_size=test_fraction)
        return train_dataset, update_dataset, test1_dataset, test2_dataset

    transform = None

    if transform_name == 'resample_metric_features':
        transform = lambda dataset, fraction : resample_metric_features(
            dataset=dataset,
            seed=_seed, 
            fraction=fraction, 
            magnitude=transform_parameters['magnitude'],
            metric_features=transform_parameters['metric_features'])

    if transform_name == 'shift_metric_features':
        transform = lambda dataset, fraction : shift_metric_features(
            dataset=dataset,
            seed=_seed,
            fraction=fraction,
            magnitude=transform_parameters['magnitude'],
            categorical_features=transform_parameters['categorical_features'])

    if transform_name == 'rotate_metric_features':
        transform = lambda dataset, fraction : rotate_metric_features(
            dataset=dataset,
            seed=_seed,
            fraction=fraction,
            magnitude=transform_parameters['magnitude'],
            categorical_features=transform_parameters['categorical_features'])

    if transform_name == 'regression_category_drift':
        transform = lambda dataset, fraction : regression_category_drift(
            dataset=dataset,
            seed=_seed,
            fraction=fraction,
            magnitude=transform_parameters['magnitude'],
            categorical_features=transform_parameters['categorical_features'],
            target=transform_parameters['target'])

    if transform != None:
        raise NotImplementedError(transform_name + ' is not implemented.')

    train_dataset, test1_dataset = train_test_split(dataset, test_size=test_fraction)
    update_dataset = transform(train_dataset, transform_parameters['fraction'])
    test2_dataset = transform(test1_dataset, 1.0)
    return train_dataset, update_dataset, test1_dataset, test2_dataset


@ex.main
def my_main(_run, dataset_name, verbose):
    results = {}

    original_dataset = read_dataset()
    train_dataset, update_dataset, test1_dataset, test2_dataset = transform_dataset(dataset=original_dataset)
    baseline_dataset = pd.concat([train_dataset, update_dataset])
    test_dataset = pd.concat([test1_dataset, test2_dataset])
    
    _run.run_logger.info("Baseline training started")
    baseline_model = train(dataset_name, baseline_dataset, verbose=verbose, run=_run)
    elapsed_time = baseline_model.wait_to_complete(clean=False)
    results['baseline'] = {}
    results['baseline']['config'] = baseline_model.get_config()
    results['baseline']['hyperparameters'] = baseline_model.get_hyperparameters()
    results['baseline']['elapsed_time'] = elapsed_time
    baseline_model.clean()
    _run.run_logger.info("Baseline training completed")

    dependent_variable = baseline_model.dependent_variable

    _run.run_logger.info("Initial training started")
    trained_model = train(dataset_name, train_dataset, verbose=verbose, run=_run)
    elapsed_time = trained_model.wait_to_complete(clean=False)
    results['trained_model'] = {}
    results['trained_model']['config'] = trained_model.get_config()
    results['trained_model']['hyperparameters'] = trained_model.get_hyperparameters()
    results['trained_model']['elapsed_time'] = elapsed_time
    trained_model.clean()
    _run.run_logger.info("Initial training completed")

    _run.run_logger.info("Update started")
    updated_model = update(dataset_name, update_dataset, trained_model, verbose=verbose, run=_run)
    elapsed_time = updated_model.wait_to_complete(clean=False)
    results['updated_model'] = {}
    results['updated_model']['config'] = updated_model.get_config()
    results['updated_model']['hyperparameters'] = updated_model.get_hyperparameters()
    results['updated_model']['elapsed_time'] = elapsed_time
    updated_model.clean()
    _run.run_logger.info("Update completed")

    baseline_eval = evaluate(dataset_name, test_dataset, baseline_model, verbose=verbose)
    baseline_eval.wait_to_complete()

    trained_model_eval = evaluate(dataset_name, test_dataset, trained_model, verbose=verbose)
    trained_model_eval.wait_to_complete()

    updated_model_eval = evaluate(dataset_name, test_dataset, updated_model, verbose=verbose)
    updated_model_eval.wait_to_complete()

    scores = {}

    if baseline_model.is_regression():
        y_true = [y for y in test_dataset[dependent_variable]]
        scores = compute_regression_metrics(y_true,
                                            baseline_eval.get_predictions(),
                                            trained_model_eval.get_predictions(),
                                            updated_model_eval.get_predictions())
    elif baseline_model.is_classification():
        y_true = [str(y) for y in test_dataset[dependent_variable]]
        scores = compute_classification_metrics(y_true,
                                                baseline_eval.get_predictions(),
                                                trained_model_eval.get_predictions(),
                                                updated_model_eval.get_predictions())
    else:
        _run.run_logger.warning(
            "Job is neither regression nor classification. No metric scores are available.")
    return always_merger.merge(results, scores)


if __name__ == '__main__':
    run = ex.run_commandline()
    run_data_path = experiment_data_path / str(run._id)
    push2es(data_path=run_data_path, name=experiment_name)
    shutil.rmtree(run_data_path)
