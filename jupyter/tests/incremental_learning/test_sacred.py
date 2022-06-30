# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.run import Run

from incremental_learning.config import datasets_dir
from incremental_learning.job import evaluate, train, update
from incremental_learning.storage import download_dataset


class TestSacred(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_name = 'test_regression'
        download_successful = download_dataset(self.dataset_name)
        if download_successful == False:
            self.fail("Dataset is not available")
        self.dataset = pd.read_csv(
            datasets_dir / '{}.csv'.format(self.dataset_name))

    def test_seed(self) -> None:
        def run_tests(dataset_name, dataset, _run) -> None:
            # make sure results are identical for the same seed
            job1 = train(dataset_name, dataset, verbose=False, run=_run)
            success = job1.wait_to_complete()
            self.assertTrue(success)
            prediction1 = job1.get_predictions()

            job2 = train(dataset_name, dataset, verbose=False, run=_run)
            success = job2.wait_to_complete()
            self.assertTrue(success)
            prediction2 = job2.get_predictions()

            self.assertTrue(np.array_equal(prediction1, prediction2))

            # change seed and make sure results are different
            _run.config['seed'] = 120
            job3 = train(dataset_name, dataset, verbose=False, run=_run)
            success = job3.wait_to_complete()
            self.assertTrue(success)
            prediction3 = job3.get_predictions()
            self.assertFalse(np.array_equal(prediction1, prediction3))

        ex = Experiment('test_seed')
        ex.add_config(seed=100, dataset_name=self.dataset_name,
                      dataset=self.dataset)
        ex.main(run_tests)
        ex.run()


if __name__ == '__main__':
    unittest.main()
