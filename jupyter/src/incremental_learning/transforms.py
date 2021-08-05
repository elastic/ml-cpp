# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import math
from pandas import DataFrame
import random

def metric_features(categorical_features : list,
                    data_frame : DataFrame):

    result = {x for x in data_frame.columns}
    result.difference({x for x in categorical_features})
    return list(result)

def matches_columns(features : list,
                    data_frame : DataFrame):
    return [x in features for x in data_frame.columns].count(True) != len(features)


def censor_feature_ranges(seed : int,
                          features : list,
                          data_frame : DataFrame):
    '''
    Partition the data frame into values rows not contained and contained in
    random intervals of features.

    The intervals so that if the feature distributions are independent then
    the split is roughly 50:50. In general though one should avoid using more
    than a few features.

    Args:
        seed The random number generation seed.
        features The features to censor.

    Return:
        A pair of data frames comprising all rows not in feature intervals
        and all rows in feature intervals as the first and second elements,
        respectively.
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

    query = ''
    for feature in features:
        q_a = data_frame[feature].quantile(
                max(0.5 - 0.5 * q_interval + random.uniform(-0.05, 0.05), 0.01)
        )
        q_b = data_frame[feature].quantile(
            min(0.5 + 0.5 * q_interval + random.uniform(-0.05, 0.05), 0.99)
        )
        query += ' and ' + feature + '>=' + str(q_a) + ' and ' + feature + '<=' + str(q_b)
    query = query[5:]
    print(query)

    return data_frame.query('not(' + query + ')'), data_frame.query(query)


def random_shift(seed : int,
                 magnitude : float,
                 number : int):
    random.seed(seed)
    return [magnitude * random.uniform(-1.0, 1.0) for _ in range(number)]

def shift_metric_features(seed : int,
                          fraction : float,
                          magnitude : float,
                          categorical_features : list,
                          data_frame : DataFrame):
    '''
    Apply a random shift to the metric features in data_frame.

    The shift is uniformly sampled from magnitude * sd * [-3, 3] for each feature.

    Args:
        seed The random number generation seed.
        fraction The fraction of rows to sample which should be in the range [0, 1].
        magnitude Controls the size of the shift.
        categorical_features A list of the categorical features.
        data_frame The data to sample and shift.
    '''

    if matches_columns(categorical_features, data_frame):
        print(list(data_frame.columns), 'does not contain', categorical_features)
        return None

    features = metric_features(categorical_features, data_frame)

    sd = data_frame[features].std()
    shift = random_shift(seed, magnitude, len(features))

    data_frame = data_frame.sample(frac=fraction, random_state=seed)
    for index in range(len(data_frame.index)):
        for i, feature in enumerate(features):
            data_frame[index, feature] += 3.0 * sd[feature] * shift[i]

    return data_frame


def random_givens_rotations(seed : int,
                            magnitude : float,
                            number : int):
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
                           x : list):
    for rotation in rotations:
        ct = math.cos(rotation['theta'])
        st = math.sin(rotation['theta'])
        x[rotation['i']] = ct * x[rotation['i']] - st * x[rotation['j']]
        x[rotation['j']] = st * x[rotation['i']] + ct * x[rotation['j']]

def rotate_metric_features(seed : int,
                           fraction : float,
                           magnitude : float,
                           categorical_features : list,
                           data_frame : DataFrame):
    '''
    Downsample and apply a random rotation to the metric feature values in data_frame.

    We use number of metric features Givens rotations though random samples of the
    interval magnitude * [0, 2 * pi]. Magnitude therefore need only be in the range
    [0, 1] and should typically be small.

    Args:
        seed The random number generation seed.
        fraction The fraction of rows to sample which should be in the range [0, 1].
        magnitude Controls the size of the rotation.
        categorical_features A list of the categorical features.
        data_frame The data to sample and rotate.
    '''

    if matches_columns(categorical_features, data_frame):
        print(list(data_frame.columns), 'does not contain', categorical_features)
        return None

    features = metric_features(categorical_features, data_frame)

    centroid = data_frame[features].mean()
    rotations = random_givens_rotations(seed, len(features), magnitude)

    data_frame = data_frame.sample(frac=fraction, random_state=seed)
    for index, row in data_frame.iterrows():
        x = [row[name] - mean for name, mean in centroid.iteritems()]
        apply_givens_rotations(rotations, x)
        for i, feature in enumerate(features):
            data_frame.at[index, feature] = centroid[feature] + x[i]

    return data_frame


def regression_category_drift(seed : int,
                              fraction : float,
                              magnitude : float,
                              categorical_features : list,
                              target : str,
                              data_frame : DataFrame):
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
    '''

    if matches_columns(categorical_features, data_frame):
        print(list(data_frame.columns), 'does not contain', categorical_features)
        return None

    sd = data_frame.std(target)

    shifts = {}
    for feature in categorical_features:
        for unique in data_frame[feature].unique():
            shifts[feature + '.' + str(unique)] = 3.0 * sd * magnitude * random.uniform(-1.0, 1.0)

    data_frame = data_frame.sample(frac=fraction, random_state=seed)
    for index, row in data_frame.iterrows():
        shift = 0.0
        for feature in categorical_features:
            shift += shifts[feature + '.' + str(row[feature])]
        data_frame.at[index, target] += shift

    return data_frame