# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import unittest
import pandas
from incremental_learning.transforms import *

class TransformsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.data_frame = pandas.read_csv('../../data/datasets/boston.csv')

    def test_metric_features(self) -> None:
        '''
        Test extracting metric features from a data frame.
        '''
        expected_metric_features = ['AGE', 'B', 'CRIM', 'DIS', 'INDUS', 'LSTAT', 'MEDV', 'NOX', 'PTRATIO', 'RM', 'TAX', 'ZN']
        actual_metric_features = metric_features(['CHAS', 'RAD'], self.data_frame)
        actual_metric_features.sort()
        self.assertEqual(actual_metric_features, expected_metric_features)

    def test_partition_on_metric_ranges(self) -> None:
        '''
        Test we the split is roughly 50:50.
        '''
        partition = partition_on_metric_ranges(100, ['this feature does not exist'], self.data_frame)
        self.assertEqual(partition[0], None)
        self.assertEqual(partition[1], None)

        partition = partition_on_metric_ranges(100, ['AGE', 'DIS'], self.data_frame)
        self.assertAlmostEqual(len(partition[0].index), len(partition[1].index), delta=100)


    def test_partition_on_categories(self) -> None:
        '''
        Test we get disjoint categories in the partition and that split is roughly 50:50.
        '''
        partition = partition_on_categories(100, ['this feature does not exist'], self.data_frame)
        self.assertEqual(partition[0], None)
        self.assertEqual(partition[1], None)

        partition = partition_on_categories(100, ['CHAS', 'RAD'], self.data_frame)
        self.assertAlmostEqual(len(partition[0].index), len(partition[1].index), delta=10)

        categories = (set(), set())
        for i in range(2):
            for _, row in partition[i].iterrows():
                categories[i].add(tuple([row[feature] for feature in ['CHAS', 'RAD']]))
        self.assertGreater(len(categories[0]), 0)
        self.assertGreater(len(categories[1]), 0)
        self.assertEqual(len(categories[0].intersection(categories[1])), 0)

    def test_shift(self) -> None:
        '''
        '''

    def test_rotate_metric_features(self) -> None:
        '''
        '''

    def test_regression_category_drift(self):
        '''
        '''

if __name__ == '__main__':
    unittest.main()
