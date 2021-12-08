# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import unittest
from collections import namedtuple
from copy import copy

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

from incremental_learning.config import jobs_dir, logger
from incremental_learning.job import Job, evaluate, train, update
from incremental_learning.storage import (download_dataset, download_job,
                                          job_exists, read_dataset, upload_job)

FakeRun = namedtuple('FakeRun', ['config', 'run_logger'])


def remove_subset(dataset: pd.DataFrame, subset: pd.DataFrame):
    dataset = dataset.merge(
        subset, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
    dataset.drop(columns=['_merge'], inplace=True)
    return dataset


def compute_error(job: Job, dataset: pd.DataFrame, dataset_name: str, _run):
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


class TestIncrementalLearning(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = 1000
        test_fraction = 0.2
        training_fraction = 0.25
        self.update_fraction = 0.1
        self.verbose = False
        self.dataset_name = 'house'

        self.run = FakeRun(
            config={'analysis': {'parameters': {'seed': self.seed}}},
            run_logger=logger)

        dataset = read_dataset(self.dataset_name)

        self.train_dataset, self.test_dataset = train_test_split(
            dataset, test_size=test_fraction,  random_state=self.seed)

        self.update_num_samples = int(
            self.train_dataset.shape[0] * self.update_fraction)

        baseline_dataset = self.train_dataset.sample(
            frac=training_fraction, random_state=self.seed)

        self.baseline_model_name = 'test_multistep_learning_baseline'
        if job_exists(self.baseline_model_name, remote=False):
            path = download_job(self.baseline_model_name)
            self.baseline_model = Job.from_file(path)
        else:
            self.baseline_model = train(self.dataset_name, baseline_dataset,
                                        verbose=False, run=self.run)
            self.baseline_model.wait_to_complete()
            path = jobs_dir / self.baseline_model_name
            self.baseline_model.store(path)
        self.used_dataset = baseline_dataset.copy()

    def sample_points(self, random_state: np.random.RandomState):
        D_update = None
        success = False
        k = 2
        while not success:
            if self.update_num_samples*k > self.train_dataset.shape[0]:
                D_update = None
                break
            largest = self.train_dataset.sample(
                n=self.update_num_samples*k, random_state=random_state)

            D_update = remove_subset(largest, self.used_dataset)
            if D_update.shape[0] < self.update_num_samples:
                k += 1
                continue
            if D_update.shape[0] >= self.update_num_samples:
                D_update = D_update.sample(n=self.update_num_samples)
                success = True
        logger.info("Sampling completed after with k={}.".format(k))
        return D_update

    def test_overfitting(self) -> None:
        """Make sure the updated model does not overfit on the test error.
        """
        hyperparameters = self.baseline_model.get_hyperparameters()
        hyperparameters['retrained_tree_eta'] = [0.1, 0.4]
        random_state = np.random.RandomState(seed=self.seed)
        previous_hyperparameters = copy(hyperparameters)
        previous_job = self.baseline_model
        previous_test_error = compute_error(job=previous_job, dataset=self.test_dataset,
                                            dataset_name=self.dataset_name, _run=self.run)
        for update_step in range(10):
            update_dataset =  self.train_dataset.sample(frac=self.update_fraction) #self.sample_points(random_state)
            update_job = update(dataset=update_dataset, dataset_name=self.dataset_name,
                                original_job=previous_job, force=True, verbose=False,
                                hyperparameter_overrides=previous_hyperparameters, run=self.run)
            update_job.wait_to_complete()
            new_hyperparameters = update_job.get_hyperparameters()
            new_test_error = compute_error(job=update_job, dataset=self.test_dataset,
                                           dataset_name=self.dataset_name, _run=self.run)

            self.assertTrue(previous_test_error > new_test_error,
                            msg='Test errors comparison failed at step {}: {} < {}'
                            .format(update_step, previous_test_error, new_test_error))
            previous_job = update_job
            previous_hyperparameters = new_hyperparameters
            previous_test_error = new_test_error
