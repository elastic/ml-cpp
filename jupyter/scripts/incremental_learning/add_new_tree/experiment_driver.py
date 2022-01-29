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
import random
import string

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from deepmerge import always_merger
from pathlib2 import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.model_selection import train_test_split

from incremental_learning.config import datasets_dir, jobs_dir, logger
from incremental_learning.elasticsearch import push2es
from incremental_learning.job import evaluate, train, update, Job
from incremental_learning.storage import download_dataset, download_job, job_exists, upload_job
from incremental_learning.transforms import partition_on_metric_ranges
from incremental_learning.transforms import regression_category_drift
from incremental_learning.transforms import resample_metric_features
from incremental_learning.transforms import rotate_metric_features
from incremental_learning.transforms import shift_metric_features

experiment_name = 'incremental-training'
uid = ''.join(random.choices(string.ascii_lowercase, k=6))
experiment_data_path = Path('/tmp/'+experiment_name)
ex = Experiment(name=uid)
ex.observers.append(FileStorageObserver(experiment_data_path))
ex.logger = logger


@ex.capture
def parameters_checksum(transform_parameters, checksum_size: int = 8) -> str:
    import hashlib
    import json
    encoded_parameters = json.dumps(
        transform_parameters, sort_keys=True).encode()
    hash = hashlib.md5(encoded_parameters)
    return hash.hexdigest()[:checksum_size]


@ex.config
def my_config():
    force_update = False
    verbose = False
    dataset_name = 'ccpp'
    test_fraction = 0.2
    threads = 1


def compute_regression_metrics(y_true,
                               baseline_model_predictions,
                               trained_model_predictions,
                               updated_model_predictions):
    scores = {
        'baseline': {
            'mae': metrics.mean_absolute_error(y_true, baseline_model_predictions),
            'mse': metrics.mean_squared_error(y_true, baseline_model_predictions)
        },
        'trained_model': {
            'mae': metrics.mean_absolute_error(y_true, trained_model_predictions),
            'mse': metrics.mean_squared_error(y_true, trained_model_predictions)
        },
        'updated_model': {
            'mae': metrics.mean_absolute_error(y_true, updated_model_predictions),
            'mse': metrics.mean_squared_error(y_true, updated_model_predictions)
        },
    }
    return scores


def compute_classification_metrics(y_true,
                                   baseline_model_predictions,
                                   trained_model_predictions,
                                   updated_model_predictions):
    scores = {
        'baseline': {
            'acc': metrics.accuracy_score(y_true, baseline_model_predictions)
        },
        'trained_model': {
            'acc': metrics.accuracy_score(y_true, trained_model_predictions)
        },
        'updated_model': {
            'acc': metrics.accuracy_score(y_true, updated_model_predictions)
        },
    }

    for label in np.unique(y_true):
        scores['baseline']['precision_' + label] = \
            metrics.precision_score(
                y_true, baseline_model_predictions, pos_label=label)
        scores['trained_model']['precision_' + label] = \
            metrics.precision_score(
                y_true, trained_model_predictions, pos_label=label)
        scores['updated_model']['precision_' + label] = \
            metrics.precision_score(
                y_true, updated_model_predictions, pos_label=label)
        scores['baseline']['recall_' + label] = \
            metrics.recall_score(
                y_true, baseline_model_predictions, pos_label=label)
        scores['trained_model']['recall_' + label] = \
            metrics.recall_score(
                y_true, trained_model_predictions, pos_label=label)
        scores['updated_model']['recall_' + label] = \
            metrics.recall_score(
                y_true, updated_model_predictions, pos_label=label)

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
    random_state = np.random.RandomState(seed=_seed)
    if transform_name == 'partition_on_metric_ranges':
        train_dataset, update_dataset = partition_on_metric_ranges(
            dataset=dataset,
            seed=_seed,
            metric_features=transform_parameters['metric_features'])
        train_dataset, test1_dataset = train_test_split(
            train_dataset, test_size=test_fraction, random_state=random_state)
        update_dataset, test2_dataset = train_test_split(
            update_dataset, test_size=test_fraction, random_state=random_state)
        return train_dataset, update_dataset, test1_dataset, test2_dataset

    transform = None

    if transform_name == 'resample_metric_features':
        def transform(dataset, fraction): return resample_metric_features(
            dataset=dataset,
            seed=_seed,
            fraction=fraction,
            magnitude=transform_parameters['magnitude'],
            metric_features=transform_parameters['metric_features'])

    if transform_name == 'shift_metric_features':
        def transform(dataset, fraction): return shift_metric_features(
            dataset=dataset,
            seed=_seed,
            fraction=fraction,
            magnitude=transform_parameters['magnitude'],
            categorical_features=transform_parameters['categorical_features'])

    if transform_name == 'rotate_metric_features':
        def transform(dataset, fraction): return rotate_metric_features(
            dataset=dataset,
            seed=_seed,
            fraction=fraction,
            magnitude=transform_parameters['magnitude'],
            categorical_features=transform_parameters['categorical_features'])

    if transform_name == 'regression_category_drift':
        def transform(dataset, fraction): return regression_category_drift(
            dataset=dataset,
            seed=_seed,
            fraction=fraction,
            magnitude=transform_parameters['magnitude'],
            categorical_features=transform_parameters['categorical_features'],
            target=transform_parameters['target'])

    if transform == None:
        raise NotImplementedError(transform_name + ' is not implemented.')

    train_dataset, test1_dataset = train_test_split(
        dataset, test_size=test_fraction, random_state=random_state)
    update_dataset = transform(train_dataset, transform_parameters['fraction'])
    test2_dataset = transform(test1_dataset, 1.0)
    return train_dataset, update_dataset, test1_dataset, test2_dataset


@ex.main
def my_main(_run, _seed, dataset_name, force_update, verbose, test_fraction, transform_name):
    if 'comment' not in _run.meta_info or not _run.meta_info['comment']:
        raise RuntimeError("Specify --comment parameter for this experiment.")
    results = {}

    random_state = np.random.RandomState(seed=_seed)

    baseline_model_name = "e{}_baseline_s{}_d{}_t{}_{}".format(
        experiment_name, _seed, dataset_name, transform_name, test_fraction, parameters_checksum())

    original_dataset = read_dataset()
    train_dataset, update_dataset, test1_dataset, test2_dataset = transform_dataset(
        dataset=original_dataset)
    baseline_dataset = pd.concat([train_dataset, update_dataset])
    test_dataset = pd.concat([test1_dataset, test2_dataset])

    _run.run_logger.info("Baseline training started")
    if job_exists(baseline_model_name, remote=True):
        path = download_job(baseline_model_name)
        baseline_model = Job.from_file(path)
    else:
        baseline_model = train(dataset_name, baseline_dataset,
                               verbose=verbose, run=_run)
        elapsed_time = baseline_model.wait_to_complete()
        path = jobs_dir / baseline_model_name
        baseline_model.store(path)
        upload_job(path)
    results['baseline'] = {}
    # results['baseline']['config'] = baseline_model.get_config()
    results['baseline']['hyperparameters'] = baseline_model.get_hyperparameters()
    # results['baseline']['elapsed_time'] = elapsed_time
    # baseline_model.clean()
    _run.run_logger.info("Baseline training completed")

    dependent_variable = baseline_model.dependent_variable

    _run.run_logger.info("Initial training started")
    trained_model_name = "e{}_trained_s{}_d{}_t{}_{}".format(
        experiment_name, _seed, dataset_name, transform_name, test_fraction, parameters_checksum())
    if job_exists(trained_model_name, remote=True):
        path = download_job(baseline_model_name)
        trained_model = Job.from_file(path)
    else:
        trained_model = train(dataset_name, train_dataset,
                            verbose=verbose, run=_run)
        elapsed_time = trained_model.wait_to_complete()
        path = jobs_dir / baseline_model_name
        baseline_model.store(path)
        upload_job(path)
    results['trained_model'] = {}
    # results['trained_model']['config'] = trained_model.get_config()
    results['trained_model']['hyperparameters'] = trained_model.get_hyperparameters()
    # results['trained_model']['elapsed_time'] = elapsed_time
    # trained_model.clean()
    _run.run_logger.info("Initial training completed")

    _run.run_logger.info("Update started")
    updated_model = update(dataset_name, update_dataset, trained_model,
                           force=force_update, verbose=verbose, run=_run)
    elapsed_time = updated_model.wait_to_complete(clean=False)
    results['updated_model'] = {}
    results['updated_model']['config'] = updated_model.get_config()
    results['updated_model']['hyperparameters'] = updated_model.get_hyperparameters()
    results['updated_model']['elapsed_time'] = elapsed_time
    updated_model.clean()
    _run.run_logger.info("Update completed")

    baseline_eval = evaluate(dataset_name, test_dataset,
                             baseline_model, verbose=verbose)
    baseline_eval.wait_to_complete()

    trained_model_eval = evaluate(
        dataset_name, test_dataset, trained_model, verbose=verbose)
    trained_model_eval.wait_to_complete()

    updated_model_eval = evaluate(
        dataset_name, test_dataset, updated_model, verbose=verbose)
    updated_model_eval.wait_to_complete()

    scores = {}

    if baseline_model.is_regression():
        y_true = np.array([y for y in test_dataset[dependent_variable]])
        scores = compute_regression_metrics(y_true,
                                            baseline_eval.get_predictions(),
                                            trained_model_eval.get_predictions(),
                                            updated_model_eval.get_predictions())
    elif baseline_model.is_classification():
        y_true = np.array([str(y) for y in test_dataset[dependent_variable]])
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
    import incremental_learning.job
    ex.add_artifact(incremental_learning.job.__file__)
    push2es(data_path=run_data_path, name=experiment_name)
    shutil.rmtree(run_data_path)
