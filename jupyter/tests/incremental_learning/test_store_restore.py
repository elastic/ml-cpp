# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import logging
import os
import tempfile
import unittest
from pathlib import Path

import numpy.testing
import pandas as pd

from incremental_learning.config import (datasets_dir, es_cloud_id, jobs_dir,
                                         logger)
from incremental_learning.job import Job, evaluate, train, update
from incremental_learning.storage import download_dataset, download_job, upload_job, delete_job, job_exists


class TestJobStoreRestore(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.seed = 100
        cls.dataset_name = 'test_regression'
        download_successful = download_dataset(cls.dataset_name)
        if download_successful == False:
            cls.fail("Dataset is not available")
        cls.dataset = pd.read_csv(
            datasets_dir / '{}.csv'.format(cls.dataset_name))
        cls.job = train(cls.dataset_name, cls.dataset, verbose=False)
        cls.job.wait_to_complete(clean=False)

    def setUp(self) -> None:
        self.destination = tempfile.NamedTemporaryFile(mode='wt')
        self.path = Path(self.destination.name)
        success = self.job.store(destination=self.path)
        self.destination.file.close()
        self.assertTrue(success)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.job.clean()
        super().tearDownClass()

    def tearDown(self) -> None:
        if self.destination:
            self.destination.close()
        if hasattr(self, 'downloaded_path') and self.downloaded_path:
            self.downloaded_path.unlink()
        super().tearDown()

    def test_job_store(self):
        self.assertTrue(self.path.exists())
        self.assertTrue(self.path.is_file())
        self.assertTrue(self.path.stat().st_size > 0)

    def test_job_download_restore(self):
        job_name = 'test_regression_job'
        self.downloaded_path = download_job(job_name)
        self.assertIsNotNone(self.downloaded_path)
        self.assertTrue(self.downloaded_path.exists())

        restoredJob = Job.fromFile(self.downloaded_path)
        self.assertTrue(restoredJob.initialized)

    def test_job_store_upload(self):
        job_name = 'test_upload_job'
        success = upload_job(job_name=job_name, local_job_path=self.path)
        self.assertTrue(success)
        self.assertTrue(job_exists(job_name, remote=True))
        delete_job(job_name)
        self.assertFalse(job_exists(job_name, remote=True))

    def test_job_store_restore(self):
        restored_job = Job.fromFile(path=self.path)
        self.assertEqual(self.job, restored_job)

    def test_job_restore_evaluate(self):
        restored_job = Job.fromFile(path=self.path)

        expected_eval_job = evaluate(
            dataset=self.dataset, dataset_name=self.dataset_name, original_job=self.job, verbose=False)
        expected_eval_job.wait_to_complete()

        actual_eval_job = evaluate(
            dataset=self.dataset, dataset_name=self.dataset_name, original_job=restored_job, verbose=False)
        actual_eval_job.wait_to_complete()

        numpy.testing.assert_array_almost_equal(
            expected_eval_job.get_predictions(), actual_eval_job.get_predictions())

    def test_job_restore_update(self):
        restored_job = Job.fromFile(path=self.path)
        update_dataset = self.dataset.sample(frac=0.1, random_state=self.seed)
        expected_update_job = update(
            dataset=update_dataset, dataset_name=self.dataset_name, original_job=self.job, verbose=False)
        expected_update_job.wait_to_complete()

        actual_update_job = update(
            dataset=update_dataset, dataset_name=self.dataset_name, original_job=restored_job, verbose=False)
        actual_update_job.wait_to_complete()

        numpy.testing.assert_array_almost_equal(
            expected_update_job.get_predictions(), actual_update_job.get_predictions())
        self.assertEqual(expected_update_job.get_hyperparameters(),
                         actual_update_job.get_hyperparameters())


if __name__ == '__main__':
    # log into console output for tests
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    unittest.main()
