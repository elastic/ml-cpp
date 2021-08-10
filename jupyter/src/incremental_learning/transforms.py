# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import bisect
import math
from pandas import DataFrame
from pandas import Series
import numpy as np
import random

import pandas

def metric_features(categorical_features : list,
                    data_frame : DataFrame) -> list:

    result = {x for x in data_frame.columns}
    result = result.difference({x for x in categorical_features})
    return list(result)

def matches_columns(features : list,
                    data_frame : DataFrame) -> bool:
    return [x in features for x in data_frame.columns].count(True) != len(features)

def features_in_bounding_box(bb : list,
                             features : list,
                             row : Series) -> bool:
    for i, feature in enumerate(features):
        if row[feature] < bb[i][0] or row[feature] > bb[i][1]:
            return False
    return True

def partition_on_metric_ranges(seed : int,
                               features : list,
                               data_frame : DataFrame) -> tuple:
    '''
    Partition the data frame into values rows not contained and contained in
    random intervals of metric features.

    The intervals are chosen so that if the feature distributions are independent
    then the split is roughly 50:50. However, in general though one should avoid
    using more than a few features as it can lead to an imbalanced partition.

    Args:
        seed The random number generation seed.
        features The features to partition on.
        data_frame The data to partition.

    Return:
        A pair of data frames comprising all rows not in feature intervals and all
        rows in feature intervals as the first and second elements, respectively.
    '''

    if matches_columns(features, data_frame):
        print(list(data_frame.columns), 'does not contain', features)
        return None, None

    random.seed(seed)

    # We take the intersection of feature intervals which we expect to contain
    # points with chance P(in interval)^n. We want P(in interval)^n = 0.5 and so
    # choose the quantile range to equal math.pow(0.5, 1.0 / len(features)) on
    # average.
    q_interval = math.pow(0.5, 1.0 / len(features))

    bb = []
    for feature in features:
        q_a = max(0.5 - 0.5 * q_interval + random.uniform(-0.025, 0.025), 0.01)
        q_b = min(0.5 + 0.5 * q_interval + random.uniform(-0.025, 0.025), 0.99)
        bb.append([data_frame[feature].quantile(q_a), data_frame[feature].quantile(q_b)])

    partition = [[], []]
    for i in range(len(data_frame.index)):
        partition[features_in_bounding_box(bb, features, data_frame.iloc[i])].append(i)

    return (pandas.DataFrame([data_frame.iloc[i] for i in partition[0]]),
            pandas.DataFrame([data_frame.iloc[i] for i in partition[1]]))

class Counters(dict):
    def __missing__(self, key):
        return 0

def partition_on_categories(seed : int,
                            features : list,
                            data_frame : DataFrame) -> tuple:
    '''
    Partition the data frame into values rows matching and not matching a random
    subset of categories.

    Args:
        seed The random number generation seed.
        features The features to partition on.
        data_frame The data to partition.

    Return:
        A pair of data frames comprising comprising the partition.
    '''

    if matches_columns(features, data_frame):
        print(list(data_frame.columns), 'does not contain', features)
        return None, None

    random.seed(seed)

    # Count all distinct category tuples.
    tuple_frequencies = Counters()
    for _, row in data_frame.iterrows():
        key = tuple([row[feature] for feature in features])
        tuple_frequencies[key] += 1

    tuples = [item for item in tuple_frequencies.items()]
    tuples.sort(key=lambda x : -x[1])

    matching = set()
    total_count = 0
    for value in tuples:
        if random.uniform(0, 1) < 0.5 and total_count + value[1] < len(data_frame.index) / 2:
            matching.add(value[0])
            total_count += value[1]    

    partition = [[], []]
    for i in range(len(data_frame.index)):
        key = tuple([data_frame.iloc[i][feature] for feature in features])
        partition[0 if key in matching else 1].append(i)

    return (pandas.DataFrame([data_frame.iloc[i] for i in partition[0]]),
            pandas.DataFrame([data_frame.iloc[i] for i in partition[1]]))


def resample_metric_features(seed : int,
                             fraction : float,
                             magnitude : float,
                             features : list,
                             data_frame : DataFrame) -> DataFrame:
    '''
    Resample by randomly weighting equally spaced quantile buckets of features.

    Args:
        seed The random number generation seed.
        fraction The fraction of rows to sample which should be in the range [0, 1].
        magnitude Controls the dissimilarity of the feature distribution after resampling.
        features The features to resample on.
        data_frame The data to transform.

    Return:
        The transformed data frame.
    '''

    if matches_columns(features, data_frame):
        print(list(data_frame.columns), 'does not contain', features)
        return None
    if fraction > 1 or fraction <= 0:
        print('fraction', fraction, 'out of range (0, 1]')
        return None

    random.seed(seed)

    ranges = []
    weights = []
    for feature in features:
        ranges.append([x[1] for x in data_frame[feature].quantile([q / 10 for q in range(1, 10)]).items()])
        weights.append([0.5 + magnitude * random.uniform(-0.5, 0.5) for _ in range(10)])
    weights_normalization = [np.sum(feature_weights) for feature_weights in weights]

    probabilities = []
    normalization = 0
    for _, row in data_frame.iterrows():
        probability = 0
        for i, feature in enumerate(features):
            j = bisect.bisect_left(ranges[i], row[feature])
            probability += weights[i][j] / weights_normalization[i]
        probability /= len(features)
        probabilities.append(probability)
        normalization += probability
    for i in range(len(probabilities)):
        probabilities[i] /= normalization

    sample = np.random.choice(range(len(probabilities)), size=int(fraction * len(probabilities) + 0.5), p=probabilities)

    return pandas.DataFrame([data_frame.iloc[i] for i in sample])


def random_shift(seed : int,
                 magnitude : float,
                 number : int) -> list:
    random.seed(seed)
    return [magnitude * random.uniform(-1.0, 1.0) for _ in range(number)]

def shift_metric_features(seed : int,
                          fraction : float,
                          magnitude : float,
                          categorical_features : list,
                          data_frame : DataFrame) -> DataFrame:
    '''
    Apply a random shift to the metric features in data_frame.

    The shift is uniformly sampled from magnitude * sd * [-3, 3] for each feature.

    Args:
        seed The random number generation seed.
        fraction The fraction of rows to sample which should be in the range [0, 1].
        magnitude Controls the size of the shift.
        categorical_features A list of the categorical features.
        data_frame The data to sample and shift.

    Return:
        The transformed data frame.
    '''

    if matches_columns(categorical_features, data_frame):
        print(list(data_frame.columns), 'does not contain', categorical_features)
        return None
    if fraction > 1 or fraction <= 0:
        print('fraction', fraction, 'out of range (0, 1]')
        return None

    features = metric_features(categorical_features, data_frame)

    sd = data_frame[features].std()
    shift = random_shift(seed, magnitude, len(features))

    if fraction < 1:
        data_frame = data_frame.sample(frac=fraction, random_state=seed)
    for index, _ in data_frame.iterrows():
        for i, feature in enumerate(features):
            data_frame.loc[index, feature] += 3.0 * shift[i] * sd[feature]

    return data_frame


def random_givens_rotations(seed : int,
                            magnitude : float,
                            number : int) -> list:
    random.seed(seed)
    result = []
    while len(result) < number:
        i = int(random.uniform(0, number))
        j = int(random.uniform(0, number))
        if i != j:
            theta = 2.0 * math.pi * magnitude * random.uniform(0.0, 1.0)
            result.append({'i': i, 'j': j, 'theta': theta})
    return result

def apply_givens_rotations(rotations : list,
                           scales : list,
                           x : list) -> None:
    for rotation in rotations:
        i = rotation['i']
        j = rotation['j']
        ct = math.cos(rotation['theta'])
        st = math.sin(rotation['theta'])
        x[i] = scales[i] * (ct * x[i] / scales[i] - st * x[j] / scales[j])
        x[j] = scales[j] * (st * x[i] / scales[i] + ct * x[j] / scales[j])

def rotate_metric_features(seed : int,
                           fraction : float,
                           magnitude : float,
                           categorical_features : list,
                           data_frame : DataFrame) -> DataFrame:
    '''
    Downsample and apply a random rotation to the metric feature values in data_frame.

    We use number of metric features Givens rotations through random samples of the
    interval magnitude * [0, 2 * pi]. Magnitude therefore need only be in the range
    [0, 1] and should typically be small.

    Args:
        seed The random number generation seed.
        fraction The fraction of rows to sample which should be in the range [0, 1].
        magnitude Controls the size of the rotation.
        categorical_features A list of the categorical features.
        data_frame The data to sample and rotate.

    Return:
        The transformed data frame.
    '''

    if matches_columns(categorical_features, data_frame):
        print(list(data_frame.columns), 'does not contain', categorical_features)
        return None
    if fraction > 1 or fraction <= 0:
        print('fraction', fraction, 'out of range (0, 1]')
        return None

    features = metric_features(categorical_features, data_frame)

    centroid = data_frame[features].mean()
    sd = data_frame[features].std()
    rotations = random_givens_rotations(seed, len(features), magnitude)
    # We need to rescale features so they are of comparible magnitude before mixing.
    scales = [sd[feature] for feature in features]

    if fraction < 1:
        data_frame = data_frame.sample(frac=fraction, random_state=seed)
    for index, row in data_frame.iterrows():
        x = [row[name] - mean for name, mean in centroid.iteritems()]
        apply_givens_rotations(rotations, scales, x)
        for i, feature in enumerate(features):
            data_frame.at[index, feature] = centroid[feature] + x[i]

    return data_frame


def regression_category_drift(seed : int,
                              fraction : float,
                              magnitude : float,
                              categorical_features : list,
                              target : str,
                              data_frame : DataFrame) -> DataFrame:
    '''
    Downsample and apply a random shift to the target variable for each distinct
    category of the categorical_features feature values in data_frame.

    The shift is uniformly sampled from magnitude * sd * [-3, 3].

    Args:
        seed The random number generation seed.
        fraction The fraction of rows to sample which should be in the range [0, 1].
        magnitude Controls the size of the shift.
        categorical_features A list of the categorical features to shift.
        data_frame The data to sample and shift.

    Return:
        The transformed data frame.
    '''

    if matches_columns(categorical_features, data_frame):
        print(list(data_frame.columns), 'does not contain', categorical_features)
        return None
    if fraction > 1 or fraction <= 0:
        print('fraction', fraction, 'out of range (0, 1]')
        return None

    sd = data_frame.std(target)

    shifts = {}
    for feature in categorical_features:
        for unique in data_frame[feature].unique():
            shifts[feature + '.' + str(unique)] = 3.0 * sd * magnitude * random.uniform(-1.0, 1.0)

    if fraction < 1:
        data_frame = data_frame.sample(frac=fraction, random_state=seed)
    for index, row in data_frame.iterrows():
        shift = 0.0
        for feature in categorical_features:
            shift += shifts[feature + '.' + str(row[feature])]
        data_frame.at[index, target] += shift

    return data_frame