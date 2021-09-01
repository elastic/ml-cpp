#!/usr/bin/env python
# coding: utf-8
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import argparse
import copy
import csv
import json
import random
from incremental_learning.config import datasets_dir
from incremental_learning.config import configs_dir
from incremental_learning.storage import download_dataset

def feature_fields(dataset_name : str):
    with open(configs_dir / '{}.json'.format(dataset_name)) as json_file:
        try:
            config = json.load(json_file)
        except:
            print('Failed reading', configs_dir / '{}.json'.format(dataset_name))
            return None

    with open(datasets_dir / '{}.csv'.format(dataset_name)) as csv_file:
        try:
            reader = csv.reader(csv_file, delimiter=',')
            field_names = next(reader)
        except:
            print('Failed reading', datasets_dir / '{}.csv'.format(dataset_name))
            return None

    target = [config['analysis']['parameters']['dependent_variable']]
    is_classification = (config['analysis']['name'] == 'classification')
    categorical_features = []
    if 'categorical_fields' in config:
        categorical_features = [name for name in config['categorical_fields']]
    metric_features = [name for name in field_names if name not in categorical_features + target]
    return target, is_classification, metric_features, categorical_features

def features():
    '''
    Get the data set target, and categorical and metric feature field names.
    '''
    result = {}
    for dataset_name in args.datasets:
        if download_dataset(dataset_name):
            fields = feature_fields(dataset_name)
            if fields != None:
                result[dataset_name] = fields
        else:
            print('Missing dataset', dataset_name)

    return result

def needs_metric_features(transform_name: str):
    '''
    These transforms require that the data set has at least one metric feature.
    (If it doesn't we don't add the "data set", "transform" combination to the
    experiments.)
    '''
    return transform_name == 'partition_on_metric_ranges' or \
           transform_name == 'resample_metric_features' or \
           transform_name == 'shift_metric_features' or \
           transform_name == 'rotate_metric_features'

def needs_categorical_features(transform_name: str):
    '''
    These transforms require that the data set has at least one categorical
    feature. (If it doesn't we don't add the "data set", "transform" combination
    to the experiments.)
    '''
    return transform_name == 'regression_category_drift'

def regression_only(transform_name: str):
    '''
    These transforms only apply to regression. (If the problem is not regression
    we don't add the "data set", "transform" combination to the experiments.)
    '''
    return transform_name == 'regression_category_drift'

def generate_parameters(transform: dict,
                        is_classification: bool,
                        target: str,
                        categorical_features: dict,
                        metric_features: dict):
    '''
    Generates the parameters for an experiment or None if it is not valid.
    '''
    if needs_metric_features(transform['transform_name']) and len(metric_features) == 0:
        return None
    if needs_categorical_features(transform['transform_name']) and len(categorical_features) == 0:
        return None
    if regression_only(transform['transform_name']) and is_classification:
        return None

    result = copy.deepcopy(transform['transform_parameters'])

    if 'fraction' in result:
        result['fraction'] = round(random.uniform(0.1, 0.9), 2)

    if 'magnitude' in result:
        result['magnitude'] = round(random.uniform(0.1, 0.9), 2)

    if 'metric_features' in result:
        if len(metric_features) > 0:
            result['metric_features'] = random.sample(
                metric_features, k=random.randint(1, min(len(metric_features), 4)))
        else:
            del result['metric_features']

    if 'categorical_features' in result:
        if len(categorical_features) > 0:
            result['categorical_features'] = categorical_features
        else:
            del result['categorical_features']

    if 'target' in result:
        result['target'] = target

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates the experiments.json file for a collection of data and transforms')
    parser.add_argument('--experiments_file', default='experiments.json', help='The experiments file to write')
    parser.add_argument('--datasets', nargs='+', default=[], help='The datasets to use')
    parser.add_argument('--transforms_file', default='transform_templates.json', help='The transforms to apply to each dataset')
    parser.add_argument('--number_random_copies', default=3, help='The number of random verions to use for each base experiment')
    parser.add_argument('--seed', default=1234567, help='The seed to use to generate experiments')
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_features = features()
    print('Dataset features', dataset_features)

    experiments = []

    with open(args.transforms_file, 'r') as transforms_file:
        for transform in json.load(transforms_file):
            print('Generating experiments for', transform)
            for _ in range(args.number_random_copies):
                for dataset_name in args.datasets:
                    if dataset_name in dataset_features:
                        target, is_classification, metric_features, categorical_features = dataset_features[dataset_name]
                        params = generate_parameters(transform=transform,
                                                        is_classification=is_classification,
                                                        target=target,
                                                        metric_features=metric_features,
                                                        categorical_features=categorical_features)
                        if params != None:
                            experiments.append({
                                'dataset_name': dataset_name,
                                'threads': 1,
                                'seed': random.randint(0, 100000000),
                                'transform_name': transform['transform_name'],
                                'transform_parameters': params
                            })

    print('There are', len(experiments), 'experiments in total')

    with open(args.experiments_file, 'w') as experiments_file:
        json.dump({'seed': args.seed, 'configurations': experiments}, experiments_file, sort_keys=True, indent=4)
