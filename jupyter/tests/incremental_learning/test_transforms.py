# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import math
import unittest
import pandas
from incremental_learning import transforms

class TransformsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.seed = 100
        self.data_frame = pandas.read_csv('../../data/datasets/boston.csv')

    def test_metric_features(self) -> None:
        '''
        Test extracting metric features from a data frame.
        '''
        expected_metric_features = ['AGE', 'B', 'CRIM', 'DIS', 'INDUS', 'LSTAT', 'MEDV', 'NOX', 'PTRATIO', 'RM', 'TAX', 'ZN']
        actual_metric_features = transforms.metric_features(['CHAS', 'RAD'], self.data_frame)
        actual_metric_features.sort()
        self.assertEqual(actual_metric_features, expected_metric_features)

    def test_partition_on_metric_ranges(self) -> None:
        '''
        Test we the split is roughly 50:50, we get a partition of the original data
        frame and seeding works as expected.
        '''
        partition = transforms.partition_on_metric_ranges(
            seed=self.seed,
            features=['this feature does not exist'],
            data_frame=self.data_frame
        )
        self.assertEqual(partition[0], None)
        self.assertEqual(partition[1], None)

        partition = transforms.partition_on_metric_ranges(
            seed=self.seed,
            features=['AGE', 'DIS'],
            data_frame=self.data_frame
        )
        self.assertAlmostEqual(
            first=len(partition[0].index),
            second=len(partition[1].index),
            delta=100
        )

        # We get a partition of the original data frame.
        self.assertEqual(len(partition[0].index) + len(partition[1].index),
                         len(self.data_frame.index))
        for i in [0, 1]:
            for j, row in partition[i].iterrows():
                self.assertEqual([value for _, value in row.iteritems()],
                                 [value for _, value in self.data_frame.loc[j].iteritems()])

        # Changing the seed results in different data while repeating with the same
        # seed produces an identical result.

        partition_with_same_seed = transforms.partition_on_metric_ranges(
            seed=self.seed,
            features=['AGE', 'DIS'],
            data_frame=self.data_frame
        )
        partition_with_different_seed = transforms.partition_on_metric_ranges(
            seed=self.seed + 1,
            features=['AGE', 'DIS'],
            data_frame=self.data_frame
        )

        for p_1, p_2, p_3 in zip(partition, partition_with_same_seed, partition_with_different_seed):
            self.assertEqual(len(p_1.index), len(p_2.index))
            self.assertNotEqual(len(p_1.index), len(p_3.index))
            for i in range(len(p_1.index)):
                self.assertEqual([column for column in p_1.iloc[i]],
                                 [column for column in p_2.iloc[i]])

    def test_partition_on_categories(self) -> None:
        '''
        Test we get disjoint categories in the partition, that the split is roughly
        50:50 and seeding works as expected.
        '''
        partition = transforms.partition_on_categories(
            seed=self.seed,
            features=['this feature does not exist'],
            data_frame=self.data_frame)
        self.assertEqual(partition[0], None)
        self.assertEqual(partition[1], None)

        partition = transforms.partition_on_categories(
            seed=self.seed,
            features=['CHAS', 'RAD'],
            data_frame=self.data_frame
        )
        self.assertAlmostEqual(
            first=len(partition[0].index),
            second=len(partition[1].index),
            delta=10
        )

        # We get a partition of the original data frame.
        self.assertEqual(len(partition[0].index) + len(partition[1].index),
                         len(self.data_frame.index))
        for i in [0, 1]:
            for j, row in partition[i].iterrows():
                self.assertEqual([value for _, value in row.iteritems()],
                                 [value for _, value in self.data_frame.loc[j].iteritems()])

        categories = (set(), set())
        for i in range(2):
            for _, row in partition[i].iterrows():
                categories[i].add(tuple([row[feature] for feature in ['CHAS', 'RAD']]))
        self.assertGreater(len(categories[0]), 0)
        self.assertGreater(len(categories[1]), 0)
        self.assertEqual(len(categories[0].intersection(categories[1])), 0)

        # Changing the seed results in different data while repeating with the same
        # seed produces an identical result.

        partition_with_same_seed = transforms.partition_on_categories(
            seed=self.seed,
            features=['CHAS', 'RAD'],
            data_frame=self.data_frame
        )
        partition_with_different_seed = transforms.partition_on_categories(
            seed=self.seed + 1,
            features=['CHAS', 'RAD'],
            data_frame=self.data_frame
        )

        for p_1, p_2, p_3 in zip(partition, partition_with_same_seed, partition_with_different_seed):
            self.assertEqual(len(p_1.index), len(p_2.index))
            self.assertNotEqual(len(p_1.index), len(p_3.index))
            for i in range(len(p_1.index)):
                self.assertEqual([column for column in p_1.iloc[i]],
                                 [column for column in p_2.iloc[i]])

    def test_resample_metric_features(self) -> None:
        '''
        Test that the sample size is correctly controlled by fraction.
        '''
        resampled_data_frame = transforms.resample_metric_features(
            seed=self.seed,
            fraction=0.2,
            magnitude=0.9,
            features=['this feature does not exist'],
            data_frame=self.data_frame
        )
        self.assertEqual(resampled_data_frame, None)

        resampled_data_frame = transforms.resample_metric_features(
            seed=self.seed,
            fraction=0.2,
            magnitude=0.9,
            features=['AGE', 'MEDV'],
            data_frame=self.data_frame
        )
        self.assertAlmostEqual(
            first=len(resampled_data_frame.index),
            second=0.2 * len(self.data_frame),
            delta=2
        )

    def test_shift_metric_features(self) -> None:
        '''
        Test that shifts and sample size are controlled by magnitude and fraction correctly.
        '''
        shifts = transforms.random_shift(self.seed, 0.1, 10)
        self.assertEqual(len(shifts), 10)
        for shift in shifts:
            self.assertLess(abs(shift), 0.1)

        shifted_data_frame = self.data_frame.copy(deep=True)
        shifted_data_frame = transforms.shift_metric_features(
            seed=self.seed,
            fraction=0.2,
            magnitude=0.1,
            categorical_features=['CHAS', 'RAD'],
            data_frame=shifted_data_frame
        )
        self.assertAlmostEqual(
            first=len(shifted_data_frame.index),
            second=0.2 * len(self.data_frame),
            delta=2
        )

        shifted_data_frame = self.data_frame.copy(deep=True)
        shifted_data_frame = transforms.shift_metric_features(
            seed=self.seed,
            fraction=1.0,
            magnitude=0.1,
            categorical_features=['CHAS', 'RAD'],
            data_frame=shifted_data_frame
        )
        self.assertEqual(len(shifted_data_frame.index), len(self.data_frame.index))

        features = transforms.metric_features(['CHAS', 'RAD'], self.data_frame)
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
        Test that rotations and sample size are controlled by magnitude and fraction
        correctly and some invariants of the transform.
        '''

        rotations = transforms.random_givens_rotations(seed=self.seed, magnitude=0.1, dimension=4)
        self.assertEqual(len(rotations), 3)
        unique_rotations = set()
        for rotation in rotations:
            self.assertNotEqual(rotation['i'], rotation['j'])
            self.assertLess(rotation['i'], rotation['j'])
            self.assertLess(rotation['theta'], 0.1 * 2.0 * math.pi)
            unique_rotations.add((rotation['i'], rotation['j']))
        self.assertEqual(len(unique_rotations), 3)

        x = [1, 0]
        transforms.apply_givens_rotations(
            rotations=[{'i' : 0, 'j' : 1, 'theta' : 0.5 * math.pi}], scales=[1, 1], x=x
        )
        self.assertAlmostEqual(first=x[0], second=0, delta=1e-8)
        self.assertAlmostEqual(first=x[1], second=1, delta=1e-8)

        rotated_data_frame = self.data_frame.copy(deep=True)
        rotated_data_frame = transforms.rotate_metric_features(
            seed=self.seed,
            fraction=0.2,
            magnitude=0.01,
            categorical_features=['CHAS', 'RAD'],
            data_frame=rotated_data_frame
        )
        self.assertAlmostEqual(
            first=len(rotated_data_frame.index),
            second=0.2 * len(self.data_frame),
            delta=2
        )

        features = transforms.metric_features(['CHAS', 'RAD'], self.data_frame)

        # Check that categorical features are preserved and metric features values
        # are close on the order of the feature standard deviation.
        selected = []
        sd = self.data_frame.std()
        for i, row in rotated_data_frame.iterrows():
            for feature in ['CHAS', 'RAD']:
                self.assertEqual(row[feature], self.data_frame.loc[i, feature])
            for feature in features:
                self.assertAlmostEqual(row[feature], self.data_frame.loc[i, feature], delta=sd[feature])
            selected.append(i)

        # Check data centroid is similar.
        rotated_centroid = rotated_data_frame.mean()
        centroid = self.data_frame.loc[selected].mean()
        for mean, rotated_mean in zip(centroid.items(), rotated_centroid.items()):
            self.assertAlmostEqual(first=1, second=rotated_mean[1] / mean[1], delta=0.02) # 2 %

    def test_regression_category_drift(self):
        '''
        Test that target difference and sample size are controlled by magnitude and
        fraction correctly.
        '''

        drifted_data_frame = transforms.regression_category_drift(
            seed=self.seed,
            fraction=0.2,
            magnitude=0.1,
            categorical_features=['this feature does not exist'],
            target='MEDV',
            data_frame=self.data_frame
        )
        self.assertEqual(drifted_data_frame, None)

        drifted_data_frame = transforms.regression_category_drift(
            seed=self.seed,
            fraction=0.2,
            magnitude=0.1,
            categorical_features=['CHAS', 'RAD'],
            target='this feature does not exist',
            data_frame=self.data_frame
        )
        self.assertEqual(drifted_data_frame, None)

        drifted_data_frame = self.data_frame.copy(deep=True)
        drifted_data_frame = transforms.regression_category_drift(
            seed=self.seed,
            fraction=0.2,
            magnitude=0.1,
            categorical_features=['CHAS'],
            target='MEDV',
            data_frame=drifted_data_frame
        )
        self.assertAlmostEqual(
            first=len(drifted_data_frame.index),
            second=0.2 * len(self.data_frame),
            delta=2
        )

        sd = self.data_frame['MEDV'].std()

        for i, row in drifted_data_frame.iterrows():
            for feature in self.data_frame.columns:
                if feature == 'MEDV':
                    self.assertAlmostEqual(
                        first=row[feature],
                        second=self.data_frame.loc[i, feature],
                        delta=3.0 * sd
                    )
                else:
                    self.assertEqual(row[feature], self.data_frame.loc[i, feature])


if __name__ == '__main__':
    unittest.main()
