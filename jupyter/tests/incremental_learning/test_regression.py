# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import pandas as pd
import unittest

from incremental_learning.config import datasets_dir
from incremental_learning.job import train, update, evaluate, encode
from incremental_learning.storage import download_dataset

class TestUpdateAndEvaluate(unittest.TestCase):

    def setUp(self) -> None:
        self.seed = 100
        self.dataset_name = 'test_regression'
        download_successful = download_dataset(self.dataset_name)
        self.assertTrue(download_successful, "Failed downloading dataset.")
        self.dataset = pd.read_csv(datasets_dir / '{}.csv'.format(self.dataset_name))

    def test_encode_train(self) -> None:
        job1 = encode(self.dataset_name, self.dataset, verbose=False)
        success = job1.wait_to_complete()
        self.assertTrue(success)
        print("Model", job1.model)
        self.assertTrue(job1.model != '')

        job2 = train(self.dataset_name, self.dataset.sample(frac=0.8), 
                     encode_job=job1, verbose=False)
        success = job2.wait_to_complete()
        self.assertTrue(success)


    def test_train_update(self) -> None:
        job1 = train(self.dataset_name, self.dataset, verbose=False)
        success = job1.wait_to_complete()
        self.assertTrue(success)
        self.assertEqual(job1.get_predictions().shape[0], 100)

        job2 = update(self.dataset_name, self.dataset, job1, verbose=False)
        success = job2.wait_to_complete()
        self.assertTrue(success)
        self.assertEqual(job2.get_predictions().shape[0], 100)

    def test_train_evaluate(self):
        job1 = train(self.dataset_name, self.dataset, verbose=False)
        success = job1.wait_to_complete()
        self.assertTrue(success)

        job2 = evaluate(self.dataset_name, self.dataset, job1, verbose=False)
        success = job2.wait_to_complete()
        self.assertTrue(success)
        self.assertEqual(job2.get_predictions().shape[0], 100)

if __name__ == '__main__':
    unittest.main()