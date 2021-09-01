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
import numpy as np
from pandas import DataFrame
from pandas import Series
from typing import List
from typing import Set
import random

import pandas

def metric_features(categorical_features : List[str],
                    dataset : DataFrame) -> List[str]:

    return [x for x in dataset.columns if x not in categorical_features]

def matches_columns(features : List[str],
                    dataset : DataFrame) -> bool:
    return [x in features for x in dataset.columns].count(True) != len(features)

def features_in_bounding_box(bb : List[float],
                             metric_features : List[str],
                             row : Series) -> bool:
    for i, feature in enumerate(metric_features):
        if row[feature] < bb[i][0] or row[feature] > bb[i][1]:
            return False
    return True

def generate_partition_on_metric_ranges(bb : List[float],
                                        metric_features : List[str],
                                        dataset : DataFrame,
                                        subset : bool) -> int:
    for i in range(len(dataset.index)):
        if features_in_bounding_box(bb, metric_features, dataset.iloc[i]) == subset:
            yield dataset.iloc[i]

def partition_on_metric_ranges(seed : int,
                               metric_features : List[str],
                               dataset : DataFrame) -> tuple:
    '''
    Partition the data frame into values rows not contained and contained in
    random intervals of metric features.

    The intervals are chosen so that if the feature distributions are independent
    then the split is roughly 50:50. However, in general though one should avoid
    using more than a few features as it can lead to an imbalanced partition.

    Args:
        seed The random number generation seed.
        metric_features The features to partition on.
        dataset The data to partition.

    Return:
        A pair of data frames comprising all rows not in feature intervals and all
        rows in feature intervals as the first and second elements, respectively.
    '''

    if matches_columns(metric_features, dataset):
        print(list(dataset.columns), 'does not contain', metric_features)
        return None, None

    numeric_types = dataset.select_dtypes(include='number').columns
    metric_features = [feature for feature in metric_features if feature in numeric_types]
    if len(metric_features) == 0:
        return None, None

    random.seed(seed)

    # We take the intersection of feature intervals which we expect to contain
    # points with chance P(in interval)^n. We want P(in interval)^n = 0.5 and so
    # choose the quantile range to equal math.pow(0.5, 1.0 / len(metric_features))
    # on average.
    q_interval = math.pow(0.5, 1.0 / len(metric_features))

    bb = []
    for feature in metric_features:
        q_a = max(0.5 - 0.5 * q_interval + random.uniform(-0.025, 0.025), 0.01)
        q_b = min(0.5 + 0.5 * q_interval + random.uniform(-0.025, 0.025), 0.99)
        bb.append([dataset[feature].quantile(q_a), dataset[feature].quantile(q_b)])

    return (pandas.DataFrame(generate_partition_on_metric_ranges(bb, metric_features, dataset, True)),
            pandas.DataFrame(generate_partition_on_metric_ranges(bb, metric_features, dataset, False)))

class Counters(dict):
    def __missing__(self, key):
        return 0

def generate_partition_on_categories(categorical_features : List[str],
                                     matching : Set[str],
                                     dataset : DataFrame,
                                     subset : bool) -> int:
    for i in range(len(dataset.index)):
        key = tuple([dataset.iloc[i][feature] for feature in categorical_features])
        if (key in matching) == subset:
            yield dataset.iloc[i]

def partition_on_categories(seed : int,
                            categorical_features : List[str],
                            dataset : DataFrame) -> tuple:
    '''
    Partition the data frame into values rows matching and not matching a random
    subset of categories.

    Args:
        seed The random number generation seed.
        categorical_features The features to partition on.
        dataset The data to partition.

    Return:
        A pair of data frames comprising comprising the partition.
    '''

    if matches_columns(categorical_features, dataset):
        print(list(dataset.columns), 'does not contain', categorical_features)
        return None, None

    random.seed(seed)

    # Count all distinct category tuples.
    tuple_frequencies = Counters()
    for _, row in dataset.iterrows():
        key = tuple([row[feature] for feature in categorical_features])
        tuple_frequencies[key] += 1

    tuples = [item for item in tuple_frequencies.items()]
    tuples.sort(key=lambda x : -x[1])

    matching = set()
    total_count = 0
    for value in tuples:
        if random.uniform(0, 1) < 0.5 and total_count + value[1] < len(dataset.index) / 2:
            matching.add(value[0])
            total_count += value[1]    

    return (pandas.DataFrame(generate_partition_on_categories(categorical_features, matching, dataset, True)),
            pandas.DataFrame(generate_partition_on_categories(categorical_features, matching, dataset, False)))


def resample_metric_features(seed : int,
                             fraction : float,
                             magnitude : float,
                             metric_features : List[str],
                             dataset : DataFrame) -> DataFrame:
    '''
    Resample by randomly weighting equally spaced quantile buckets of features.

    Args:
        seed The random number generation seed.
        fraction The fraction of rows to sample which should be in the range [0, 1].
        magnitude Controls the dissimilarity of the feature distribution after resampling.
        metric_features The features to resample on.
        dataset The data to transform.

    Return:
        The resampled data frame.
    '''

    if matches_columns(metric_features, dataset):
        print(list(dataset.columns), 'does not contain', metric_features)
        return None
    if fraction > 1 or fraction <= 0:
        print('fraction', fraction, 'out of range (0, 1]')
        return None

    numeric_types = dataset.select_dtypes(include='number').columns
    metric_features = [feature for feature in metric_features if feature in numeric_types]
    if len(metric_features) == 0:
        return None

    random.seed(seed)

    ranges = []
    weights = []
    quantiles = dataset.quantile([q / 10 for q in range(1, 10)])
    for feature in metric_features:
        if feature in quantiles.columns:
            ranges.append(quantiles[feature].tolist())
            weights.append([0.5 + magnitude * random.uniform(-0.5, 0.5) for _ in range(11)])
    weights_normalization = [np.sum(feature_weights) for feature_weights in weights]

    probabilities = []
    normalization = 0
    for _, row in dataset.iterrows():
        probability = 1
        for i, feature in enumerate(metric_features):
            j = bisect.bisect_left(ranges[i], row[feature])
            probability *= weights[i][j] / weights_normalization[i]
        probability /= len(metric_features)
        probabilities.append(probability)
        normalization += probability
    for i in range(len(probabilities)):
        probabilities[i] /= normalization

    sample = np.random.choice(
        range(len(probabilities)),
        size=int(fraction * len(probabilities) + 0.5),
        p=probabilities
    )

    return pandas.DataFrame(dataset.iloc[i] for i in sample)


def random_shift(seed : int,
                 magnitude : float,
                 number : int) -> list:
    random.seed(seed)
    return [magnitude * random.uniform(-1.0, 1.0) for _ in range(number)]

def shift_metric_features(seed : int,
                          fraction : float,
                          magnitude : float,
                          categorical_features : List[str],
                          dataset : DataFrame) -> DataFrame:
    '''
    Apply a random shift to the metric features in dataset.

    The shift is uniformly sampled from magnitude * sd * [-3, 3] for each feature.

    Args:
        seed The random number generation seed.
        fraction The fraction of rows to sample which should be in the range [0, 1].
        magnitude Controls the size of the shift.
        categorical_features A list of the categorical features.
        dataset The data to sample and shift.

    Return:
        The transformed data frame.
    '''

    if matches_columns(categorical_features, dataset):
        print(list(dataset.columns), 'does not contain', categorical_features)
        return None
    if fraction > 1 or fraction <= 0:
        print('fraction', fraction, 'out of range (0, 1]')
        return None

    features = metric_features(categorical_features, dataset)

    numeric_types = dataset.select_dtypes(include='number').columns
    features = [feature for feature in features if feature in numeric_types]
    if len(features) == 0:
        return None

    sd = dataset[features].std()
    shift = random_shift(seed, magnitude, len(features))

    result = dataset.sample(frac=fraction, random_state=seed) if fraction < 1 else dataset.copy(deep=True)
    for i, _ in result.iterrows():
        for j, feature in enumerate(features):
            result.loc[i, feature] += 3.0 * shift[j] * sd[feature]

    return result


def random_givens_rotations(seed : int,
                            magnitude : float,
                            dimension : int) -> list:
    random.seed(seed)

    result = []
    unique_rotations = set()
    while len(result) + 1 < dimension:
        i = int(random.uniform(0, dimension))
        j = int(random.uniform(0, dimension))
        if i > j:
            i, j = j, i
        if i != j and (i, j) not in unique_rotations:
            theta = 2.0 * math.pi * magnitude * random.uniform(0.0, 1.0)
            result.append({'i': i, 'j': j, 'theta': theta})
            unique_rotations.add((i, j))

    return result

def apply_givens_rotations(rotations : list,
                           scales : List[float],
                           x : List[float]) -> None:
    for rotation in rotations:
        i, j = rotation['i'], rotation['j']
        xi, xj = x[i], x[j]
        ct, st = math.cos(rotation['theta']), math.sin(rotation['theta'])
        x[i] = scales[i] * (ct * xi / scales[i] - st * xj / scales[j])
        x[j] = scales[j] * (st * xi / scales[i] + ct * xj / scales[j])

def rotate_metric_features(seed : int,
                           fraction : float,
                           magnitude : float,
                           categorical_features : List[str],
                           dataset : DataFrame) -> DataFrame:
    '''
    Downsample and apply a random rotation to the metric feature values in dataset.

    We use number of metric features Givens rotations through random samples of the
    interval magnitude * [0, 2 * pi]. Magnitude therefore need only be in the range
    [0, 1] and should typically be small.

    Args:
        seed The random number generation seed.
        fraction The fraction of rows to sample which should be in the range [0, 1].
        magnitude Controls the size of the rotation.
        categorical_features A list of the categorical features.
        dataset The data to sample and rotate.

    Return:
        The transformed data frame.
    '''

    if matches_columns(categorical_features, dataset):
        print(list(dataset.columns), 'does not contain', categorical_features)
        return None
    if fraction > 1 or fraction <= 0:
        print('fraction', fraction, 'out of range (0, 1]')
        return None

    features = metric_features(categorical_features, dataset)
    features.sort()

    numeric_types = dataset.select_dtypes(include='number').columns
    features = [feature for feature in features if feature in numeric_types]
    if len(features) == 0:
        return None

    centroid = dataset[features].mean()
    rotations = random_givens_rotations(seed, magnitude, len(features))
    # We need to rescale features so they are of comparible magnitude before mixing.
    sd = dataset[features].std()
    scales = [sd[feature] for feature in features]

    result = dataset.sample(frac=fraction, random_state=seed) if fraction < 1 else dataset.copy(deep=True)
    for i, row in result.iterrows():
        x = [row[feature] - centroid[feature] for feature in features]
        apply_givens_rotations(rotations, scales, x)
        for j, feature in enumerate(features):
            result.loc[i, feature] = centroid[feature] + x[j]

    return result


def regression_category_drift(seed : int,
                              fraction : float,
                              magnitude : float,
                              categorical_features : List[str],
                              target : str,
                              dataset : DataFrame) -> DataFrame:
    '''
    Downsample and apply a random shift to the target variable for each distinct
    category of the categorical_features feature values in dataset.

    The shift is uniformly sampled from magnitude * sd * [-3, 3].

    Args:
        seed The random number generation seed.
        fraction The fraction of rows to sample which should be in the range [0, 1].
        magnitude Controls the size of the shift.
        categorical_features A list of the categorical features to shift.
        dataset The data to sample and shift.

    Return:
        The transformed data frame.
    '''

    if matches_columns(categorical_features, dataset):
        print(list(dataset.columns), 'does not contain', categorical_features)
        return None
    if matches_columns([target], dataset):
        print(list(dataset.columns), 'does not contain', '"' + target + '"')
        return None
    if fraction > 1 or fraction <= 0:
        print('fraction', fraction, 'out of range (0, 1]')
        return None

    sd = dataset[target].std()

    shifts = {feature : {} for feature in categorical_features}
    for feature in categorical_features:
        for unique in dataset[feature].unique():
            shifts[feature][unique] = 3.0 * sd * magnitude * random.uniform(-1.0, 1.0)

    result = dataset.sample(frac=fraction, random_state=seed) if fraction < 1 else dataset.copy(deep=True)
    for i, row in result.iterrows():
        shift = 0.0
        for feature in categorical_features:
            shift += shifts[feature][row[feature]]
        result.loc[i, target] += shift

    return result