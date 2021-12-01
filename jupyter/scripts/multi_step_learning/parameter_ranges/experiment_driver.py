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

import diversipy
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from deepmerge import always_merger
from incremental_learning.config import datasets_dir, jobs_dir, logger
from incremental_learning.elasticsearch import push2es
from incremental_learning.job import Job, evaluate, train, update
from incremental_learning.storage import (download_dataset, download_job,
                                          job_exists, upload_job)
from incremental_learning.transforms import (partition_on_metric_ranges,
                                             regression_category_drift,
                                             resample_metric_features,
                                             rotate_metric_features,
                                             shift_metric_features)
from incremental_learning.trees import Forest
from pathlib2 import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.model_selection import train_test_split

experiment_name = 'multi-step-sampling'
uid = ''.join(random.choices(string.ascii_lowercase, k=6))
experiment_data_path = Path('/tmp/'+experiment_name)
ex = Experiment(name=uid)
ex.observers.append(FileStorageObserver(experiment_data_path))
ex.logger = logger


@ex.config
def my_config():
    force_update = True
    verbose = False
    dataset_name = 'house'
    test_fraction = 0.1
    training_fraction = 0.1
    update_fraction = 0.1
    threads = 8
    update_steps = 10

    analysis_parameters = {'parameters':
                           {'tree_topology_change_penalty': 0.0,
                            'prediction_change_cost': 0.0,
                            'data_summarization_fraction': 1.0,
                            'early_stopping_enabled': False,
                            'max_optimization_rounds_per_hyperparameter': 10,
                            'downsample_factor': 1.0}}
    sampling_mode = 'nlargest'
    n_largest_multiplier = 1
    submit = True


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


def remove_subset(dataset, subset):
    dataset = dataset.merge(
        subset, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
    dataset.drop(columns=['_merge'], inplace=True)
    return dataset


@ex.capture
def sample_points(total_dataset, update_num_samples, used_dataset, random_state, _run):
    D_update = None
    success = False
    k = 2
    while not success:
        if update_num_samples*k > total_dataset.shape[0]:
            D_update = None
            break
        largest = total_dataset.sample(
            n=update_num_samples*k, weights='indicator', random_state=random_state)

        largest.drop(columns=['indicator'], inplace=True)
        D_update = remove_subset(largest, used_dataset)
        if D_update.shape[0] < update_num_samples:
            k += 1
            continue
        if D_update.shape[0] >= update_num_samples:
            D_update = D_update.sample(n=update_num_samples)
            success = True
    _run.run_logger.info("Sampling completed after with k={}.".format(k))
    return D_update


@ex.main
def my_main(_run, _seed, dataset_name, force_update, verbose, test_fraction, training_fraction, update_fraction,
            update_steps, n_largest_multiplier, analysis_parameters, sampling_mode):
    if 'comment' not in _run.meta_info or not _run.meta_info['comment']:
        raise RuntimeError("Specify --comment parameter for this experiment.")
    results = {}

    if verbose:
        ex.logger.setLevel("DEBUG")

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

    _run.run_logger.info(
        "Baseline training started for {}".format(baseline_model_name))
    if job_exists(baseline_model_name, remote=True):
        path = download_job(baseline_model_name)
        baseline_model = Job.from_file(path)
    else:
        baseline_model = train(dataset_name, baseline_dataset,
                               verbose=verbose, run=_run)
        elapsed_time = baseline_model.wait_to_complete()
        path = jobs_dir / baseline_model_name
        baseline_model.store(path)
        upload_job(path, True)
    results['baseline'] = {}
    results['baseline']['hyperparameters'] = baseline_model.get_hyperparameters()
    results['baseline']['forest_statistics'] = get_forest_statistics(
        baseline_model.get_model_definition())
    results['baseline']['train_error'] = compute_error(
        baseline_model, train_dataset)
    results['baseline']['test_error'] = compute_error(
        baseline_model, test_dataset)
    _run.run_logger.info("Baseline training completed")
    used_dataset = baseline_dataset.copy()

    previous_model = baseline_model
    fraction_of_train = 0.1
    for step in range(update_steps):
        _run.run_logger.info("Update step {} started".format(step))

        if sampling_mode == 'nlargest':
            train_dataset['indicator'] = get_residuals(
                previous_model, train_dataset)

            D_update = sample_points(
                train_dataset, update_num_samples, used_dataset, random_state)
            train_dataset.drop(columns=['indicator'], inplace=True)
        else:
            D_update = train_dataset.sample(
                n=update_num_samples, random_state=random_state)
        if D_update is None:
            _run.run_logger.warning(
                "Update loop interrupted. It seems that you used up all the data!")
            break
        fraction_of_train += update_fraction
        updated_model = update(dataset_name=dataset_name, dataset=D_update, original_job=previous_model,
                               force=force_update, verbose=True, run=_run)
        elapsed_time = updated_model.wait_to_complete(clean=False)
        
        _run.run_logger.debug("Hyperparameters: {}".format(
            updated_model.get_hyperparameters()))
        _run.log_scalar('updated_model.elapsed_time', elapsed_time)
        _run.log_scalar('updated_model.train_error',
                        compute_error(updated_model, train_dataset))
        _run.log_scalar('updated_model.test_error',
                        compute_error(updated_model, test_dataset))
        _run.log_scalar('training_fraction', training_fraction)
        _run.log_scalar('seed', _seed)
        used_dataset = pd.concat([used_dataset, D_update])
        _run.run_logger.info(
            "New size of used data is {}, it is {:.0f}% of training data".format(used_dataset.shape[0],
                                                                                 used_dataset.shape[0]/train_dataset.shape[0]*100))

        _run.log_scalar('updated_model.hyperparameters',
                        updated_model.get_hyperparameters())

        _run.log_scalar('run.comment', _run.meta_info['comment'])
        _run.log_scalar('run.config', _run.config)
        for k, v in get_forest_statistics(updated_model.get_model_definition()).items():
            _run.log_scalar('updated_model.forest_statistics.{}'.format(k), v)
        updated_model.clean()

        previous_model = updated_model
        _run.run_logger.info("Update step completed".format(step))

    return results


if __name__ == '__main__':
    run = ex.run_commandline()
    run_data_path = experiment_data_path / str(run._id)
    import incremental_learning.job
    ex.add_artifact(incremental_learning.job.__file__)
    if run.config['submit']:
        push2es(data_path=run_data_path, name=experiment_name)
    else:
        run.run_logger.info(
            "Experiment results were not submitted to the index.")
    shutil.rmtree(run_data_path)
