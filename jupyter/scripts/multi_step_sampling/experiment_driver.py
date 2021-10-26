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

import diversipy
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from deepmerge import always_merger
from incremental_learning.config import datasets_dir, logger, jobs_dir
from incremental_learning.elasticsearch import push2es
from incremental_learning.job import evaluate, train, update, Job
from incremental_learning.trees import Forest
from incremental_learning.storage import download_dataset, job_exists, download_job, upload_job
from incremental_learning.transforms import (partition_on_metric_ranges,
                                             regression_category_drift,
                                             resample_metric_features,
                                             rotate_metric_features,
                                             shift_metric_features)
from pathlib2 import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.model_selection import train_test_split

experiment_name = 'multi-step-sampling'
experiment_data_path = Path('/tmp/'+experiment_name)
ex = Experiment(experiment_name)
ex.observers.append(FileStorageObserver(experiment_data_path))
ex.logger = logger


@ex.config
def my_config():
    force_update = True
    verbose = False
    tag = 'valeriy, multi-step-sampling'
    dataset_name = 'house'
    test_fraction = 0.2
    training_fraction = 0.1
    update_fraction = 0.1
    threads = 8
    update_steps = 10

    analysis_parameters = {'parameters':
                           {'tree_topology_change_penalty': 0.0,
                            'prediction_change_cost': 0.0,
                            'data_summarization_fraction': 1.0}}
    sampling_mode = 'nlargest'
    n_largest_multiplier = 1


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
def compute_error(job, dataset, dataset_name, _run):
    job_eval = evaluate(dataset_name=dataset_name, dataset=dataset,
                        original_job=job, run=_run, verbose=False)
    job_eval.wait_to_complete()
    dependent_variable = job.dependent_variable
    if job.is_regression():
        y_true = np.array([y for y in dataset[dependent_variable]])
        return metrics.mean_squared_error(y_true, job_eval.get_predictions())
    elif job.is_classification():
        y_true = np.array([str(y) for y in dataset[dependent_variable]])
        return metrics.accuracy_score(y_true,
                                      job_eval.get_predictions())
    else:
        _run.run_logger.warning(
            "Job is neither regression nor classification. No metric scores are available.")
    return -1


@ex.capture
def get_residuals(job, dataset, dataset_name, _run):
    job_eval = evaluate(dataset_name=dataset_name, dataset=dataset,
                        original_job=job, run=_run, verbose=False)
    job_eval.wait_to_complete()
    predictions = job_eval.get_predictions()
    residuals = np.absolute(dataset[job_eval.dependent_variable] - predictions)
    return residuals


def get_forest_statistics(model_definition):
    trained_models = model_definition['trained_model']['ensemble']['trained_models']
    forest = Forest(trained_models)
    result = {}
    result['num_trees'] = len(forest)
    result['tree_nodes_max'] = forest.max_tree_length()
    result['tree_nodes_mean'] = np.mean(forest.tree_sizes())
    result['tree_nodes_std'] = np.std(forest.tree_sizes())
    result['tree_depth_mean'] = np.mean(forest.tree_depths())
    result['tree_depth_std'] = np.std(forest.tree_depths())
    return result


@ex.main
def my_main(_run, _seed, dataset_name, force_update, verbose, test_fraction, training_fraction, update_fraction,
            update_steps, n_largest_multiplier, analysis_parameters, sampling_mode):
    results = {}

    _run.config['analysis'] = analysis_parameters

    random_state = np.random.RandomState(seed=_seed)

    baseline_model_name = "e{}_s{}_d{}_t{}".format(
        experiment_name, _seed, dataset_name, training_fraction)

    original_dataset = read_dataset()
    train_dataset, test_dataset = train_test_split(
        original_dataset, test_size=test_fraction, random_state=random_state)
    train_dataset = train_dataset.copy()
    test_dataset = test_dataset.copy()
    baseline_dataset = train_dataset.sample(
        frac=training_fraction, random_state=random_state)
    update_num_samples = int(train_dataset.shape[0]*update_fraction)

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
    results['baseline']['hyperparameters'] = baseline_model.get_hyperparameters()
    results['baseline']['forest_statistics'] = get_forest_statistics(
        baseline_model.get_model_definition())
    _run.run_logger.info("Baseline training completed")

    previous_model = baseline_model
    results['updated_model'] = {'elapsed_time': [],
                                'train_error': [], 'test_error': [], 'hyperparameters': [], 'forest_statistics': []}
    for step in range(update_steps):
        _run.run_logger.info("Update step {} started".format(step))

        _run.run_logger.info("Sampling started")
        if sampling_mode == 'nlargest':
            train_dataset['indicator'] = get_residuals(
                previous_model, train_dataset)
            largest = train_dataset.nlargest(
                n=n_largest_multiplier*update_num_samples, columns=['indicator'])
            largest.drop(columns=['indicator'], inplace=True)
            train_dataset.drop(columns=['indicator'], inplace=True)
            du = diversipy.subset.psa_select(
                largest.to_numpy(), update_num_samples)
            D_update = pd.DataFrame(
                data=du, columns=largest.columns)
        else:
            D_update = train_dataset.sample(
                n=update_num_samples, random_state=random_state)
        _run.run_logger.info("Sampling completed")

        _run.run_logger.info("Update started")
        updated_model = update(dataset_name=dataset_name, dataset=D_update, original_job=previous_model,
                               force=force_update, verbose=verbose, run=_run)
        elapsed_time = updated_model.wait_to_complete(clean=False)
        results['updated_model']['elapsed_time'].append(elapsed_time)
        results['updated_model']['train_error'].append(compute_error(
            updated_model, train_dataset))
        results['updated_model']['test_error'].append(compute_error(
            updated_model, test_dataset))
        results['updated_model']['hyperparameters'].append(
            updated_model.get_hyperparameters())
        results['updated_model']['forest_statistics'].append(
            get_forest_statistics(updated_model.get_model_definition()))
        updated_model.clean()
        _run.run_logger.info("Update completed")

        previous_model = updated_model
        _run.run_logger.info("Update step completed".format(step))

    return results


if __name__ == '__main__':
    run = ex.run_commandline()
    run_data_path = experiment_data_path / str(run._id)
    push2es(data_path=run_data_path, name=experiment_name)
    shutil.rmtree(run_data_path)
