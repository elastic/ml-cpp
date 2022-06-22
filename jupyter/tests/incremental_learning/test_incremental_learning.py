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
import sklearn.datasets
from sklearn.model_selection import train_test_split

from incremental_learning.config import jobs_dir, logger
from incremental_learning.job import Job, evaluate, train, update
from incremental_learning.storage import (download_dataset, download_job,
                                          job_exists, upload_job)

test_hyperparameters_list = ['alpha', 'gamma', 'lambda',
                             'soft_tree_depth_limit', 'soft_tree_depth_tolerance',
                             'retrained_tree_eta']

FakeRun = namedtuple('FakeRun', ['config', 'run_logger'])


class TestIncrementalLearning(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = 1000
        n_samples = 5000
        n_features = 2
        n_informative = 2
        noise = 0.05
        test_fraction = 0.2
        training_fraction = 0.5
        self.verbose = False
        self.dataset_name = 'test_regression'

        self.run = FakeRun(
            config={'analysis': {'parameters': {'seed': self.seed,
                                                'disable_hyperparameter_scaling': True}}},
            run_logger=logger)

        download_successful = download_dataset(self.dataset_name)
        if download_successful == False:
            self.fail("Dataset is not available")
        X, y = sklearn.datasets.make_regression(n_samples=n_samples, n_features=n_features,
                                                n_informative=n_informative, noise=noise, random_state=self.seed)
        dataset = pd.DataFrame(
            np.hstack([X, y.reshape(-1, 1)]), columns=['x1', 'x2', 'y'])

        self.train_dataset, self.test_dataset = train_test_split(
            dataset, test_size=test_fraction,  random_state=self.seed)

        baseline_dataset = self.train_dataset.sample(
            frac=training_fraction, random_state=self.seed)

        self.baseline_model_name = 'test_incremental_learning_baseline'
        if job_exists(self.baseline_model_name, remote=False):
            path = download_job(self.baseline_model_name)
            self.baseline_model = Job.from_file(path)
        else:
            self.baseline_model = train(self.dataset_name, baseline_dataset,
                                        verbose=False, run=self.run)
            self.baseline_model.wait_to_complete()
            path = jobs_dir / self.baseline_model_name
            self.baseline_model.store(path)

    def test_hyperparameter_consistency(self) -> None:
        """Make sure that we get the same results when searching for hyperparameters or having and
        explicit value set.
        """
        hyperparameters = self.baseline_model.get_hyperparameters()

        for hyperparameter in test_hyperparameters_list:
            with self.subTest(msg="Testing hyperparameter", hyperparameter=hyperparameter):
                # Fix all hyperparameters except of one
                test_hyperparameters = copy(hyperparameters)
                if hyperparameter != 'retrained_tree_eta':
                    value = test_hyperparameters[hyperparameter]
                    test_hyperparameters[hyperparameter] = [value/2, value*2]
                else:
                    test_hyperparameters[hyperparameter] = [0.1, 1.0]

                update_dataset = self.train_dataset.sample(
                    frac=0.5, random_state=self.seed)
                # Run update job on the baseline model with hyperparameter search

                update_job_expected = update(dataset=update_dataset, dataset_name=self.dataset_name,
                                             original_job=self.baseline_model, force=True, verbose=True,
                                             hyperparameter_overrides=test_hyperparameters, run=self.run)
                update_job_expected.wait_to_complete()

                # Run update job on the baseline model with fixed hyperparameter
                # from the job above
                test_hyperparameters[hyperparameter] = update_job_expected.get_hyperparameters()[
                    hyperparameter]
                update_job_actual = update(dataset=update_dataset, dataset_name=self.dataset_name,
                                           original_job=self.baseline_model, force=True, verbose=True,
                                           hyperparameter_overrides=test_hyperparameters, run=self.run)
                update_job_actual.wait_to_complete()

                # the hyperparameter should have been searched
                self.assertNotEqual(update_job_expected.get_hyperparameters()[
                                    hyperparameter], 0.0, msg='Hyperparameter {} was not tuned (value 0.0)'.format(hyperparameter))
                # Compare hyperparameters
                self.assertDictEqual(update_job_expected.get_hyperparameters(
                ), update_job_actual.get_hyperparameters())
                # Compare training errors
                np.testing.assert_array_equal(update_job_expected.get_predictions(),
                                              update_job_actual.get_predictions())

                # Compare test errors
                eval_job_expected = evaluate(dataset=self.test_dataset, dataset_name=self.dataset_name,
                                             original_job=update_job_expected, verbose=False, run=self.run)
                eval_job_expected.wait_to_complete()
                eval_job_actual = evaluate(dataset=self.test_dataset, dataset_name=self.dataset_name,
                                           original_job=update_job_actual, verbose=False, run=self.run)
                eval_job_actual.wait_to_complete()
                np.testing.assert_array_equal(eval_job_expected.get_predictions(),
                                              eval_job_actual.get_predictions())

    def test_hyperparameter_monotonicity(self) -> None:
        """Make sure hyperparameter search respects given ranges."""
        hyperparameters = self.baseline_model.get_hyperparameters()
        def increase(value): return (value, value*2)
        def decrease(value): return (value/2, value)
        for hyperparameter in test_hyperparameters_list:
            for compute_range in [increase, decrease]:
                with self.subTest(msg="Testing hyperparameter decreasing sequence",
                                  hyperparameter=hyperparameter):
                    random_state = np.random.RandomState(seed=self.seed)
                    previous_hyperparameters = copy(hyperparameters)
                    previous_job = self.baseline_model
                    for update_step in range(2):
                        value = previous_hyperparameters[hyperparameter]
                        min, max = compute_range(value)
                        if hyperparameter == 'retrained_tree_eta':
                            min, max = tuple(np.clip([min, max], 0.01, 1.0))
                        previous_hyperparameters[hyperparameter] = [min, max]
                        update_dataset = self.train_dataset.sample(
                            frac=0.1, random_state=random_state)
                        update_job = update(dataset=update_dataset, dataset_name=self.dataset_name,
                                            original_job=previous_job, force=True, verbose=False,
                                            hyperparameter_overrides=previous_hyperparameters, run=self.run)
                        update_job.wait_to_complete()
                        actual_hyperparameters = update_job.get_hyperparameters()
                        self.assertTrue(
                            min <= actual_hyperparameters[hyperparameter] <= max,
                            msg='Range test for {} failed on step {}'.format(hyperparameter, update_step))
                        for key in actual_hyperparameters.keys():
                            if key not in [hyperparameter, 'previous_train_num_rows', 'previous_train_loss_gap']:
                                self.assertAlmostEqual(
                                    previous_hyperparameters[key], actual_hyperparameters[key], places=8,
                                    msg='Invariability check for {} failed on step {}'.format(key, update_step))
                        previous_job = update_job
                        previous_hyperparameters = actual_hyperparameters
