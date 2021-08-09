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
        self.seed = 100
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
        Test we the split is roughly 50:50 and seeding works as expected.
        '''
        partition = partition_on_metric_ranges(
            seed=self.seed,
            features=['this feature does not exist'],
            data_frame=self.data_frame
        )
        self.assertEqual(partition[0], None)
        self.assertEqual(partition[1], None)

        partition = partition_on_metric_ranges(
            seed=self.seed,
            features=['AGE', 'DIS'],
            data_frame=self.data_frame
        )
        self.assertAlmostEqual(
            first=len(partition[0].index),
            second=len(partition[1].index),
            delta=100
        )

        # Changing the seed results in the same data frame while repeating with the
        # same seed produces the same result.

        partition_with_same_seed = partition_on_metric_ranges(
            seed=self.seed,
            features=['AGE', 'DIS'],
            data_frame=self.data_frame
        )
        partition_with_different_seed = partition_on_metric_ranges(
            seed=self.seed + 1,
            features=['AGE', 'DIS'],
            data_frame=self.data_frame
        )

        for p_1, p_2, p_3 in zip(partition, partition_with_same_seed, partition_with_different_seed):
            self.assertEqual(len(p_1.index), len(p_2.index))
            self.assertNotEqual(len(p_1.index), len(p_3.index))
            for i in range(len(p_1.index)):
                self.assertEqual([column for column in p_1.loc[i]],
                                 [column for column in p_2.loc[i]])

    def test_partition_on_categories(self) -> None:
        '''
        Test we get disjoint categories in the partition and that split is roughly 50:50.
        '''
        partition = partition_on_categories(
            seed=self.seed,
            features=['this feature does not exist'],
            data_frame=self.data_frame)
        self.assertEqual(partition[0], None)
        self.assertEqual(partition[1], None)

        partition = partition_on_categories(self.seed, ['CHAS', 'RAD'], self.data_frame)
        self.assertAlmostEqual(
            first=len(partition[0].index),
            second=len(partition[1].index),
            delta=10
        )

        categories = (set(), set())
        for i in range(2):
            for _, row in partition[i].iterrows():
                categories[i].add(tuple([row[feature] for feature in ['CHAS', 'RAD']]))
        self.assertGreater(len(categories[0]), 0)
        self.assertGreater(len(categories[1]), 0)
        self.assertEqual(len(categories[0].intersection(categories[1])), 0)

    def test_shift_metric_features(self) -> None:
        '''
        Test that data shift and sample size are controlled by magnitude and fraction correctly.
        '''
        shifts = random_shift(self.seed, 0.1, 10)
        self.assertEqual(len(shifts), 10)
        for shift in shifts:
            self.assertLess(abs(shift), 0.1)

        shifted_data_frame = self.data_frame.copy(deep=True)
        shifted_data_frame = shift_metric_features(
            seed=self.seed,
            fraction=0.2,
            magnitude=0.1,
            categorical_features=['CHAS', 'RAD'],
            data_frame=shifted_data_frame
        )
        self.assertAlmostEqual(
            first=len(shifted_data_frame.index),
            second=0.2 * len(self.data_frame),
            delta=10
        )

        shifted_data_frame = self.data_frame.copy(deep=True)
        shifted_data_frame = shift_metric_features(
            seed=self.seed,
            fraction=1.0,
            magnitude=0.1,
            categorical_features=['CHAS', 'RAD'],
            data_frame=shifted_data_frame
        )
        self.assertEqual(len(shifted_data_frame.index), len(self.data_frame.index))

        features = metric_features(['CHAS', 'RAD'], self.data_frame)
        sd = self.data_frame[features].std()

        for i in range(len(self.data_frame.index)):
            for feature in features:
                self.assertAlmostEqual(
                    first=shifted_data_frame.loc[i, feature],
                    second=self.data_frame.loc[i, feature],
                    delta=0.1 * 3.0 * sd[feature]
                )

    def test_rotate_metric_features(self) -> None:
        '''
        '''

    def test_regression_category_drift(self):
        '''
        '''

if __name__ == '__main__':
    unittest.main()
