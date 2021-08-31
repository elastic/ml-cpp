import pandas as pd
import unittest

from incremental_learning.config import datasets_dir
from incremental_learning.job import train, update, evaluate
from incremental_learning.storage import download_dataset

class TestUpdateAndEvaluate(unittest.TestCase):

    def setUp(self) -> None:
        self.seed = 100
        self.dataset_name = 'test_regression'
        download_successful = download_dataset(self.dataset_name)
        if download_successful == False:
            self.fail("Dataset is not available")
        self.dataset = pd.read_csv(datasets_dir / '{}.csv'.format(self.dataset_name))

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