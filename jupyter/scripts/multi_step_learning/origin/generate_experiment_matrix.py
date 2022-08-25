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
from typing import List
from incremental_learning.config import datasets_dir
from incremental_learning.config import configs_dir
from incremental_learning.storage import download_dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates the experiments.json file for a collection of data and transforms')
    parser.add_argument('--experiments_file', default='experiments.json', help='The experiments file to write')
    parser.add_argument('--datasets', nargs='+', default=[], help='The datasets to use')
    parser.add_argument('--number_random_copies', default=3, help='The number of random verions to use for each base experiment', type=int)
    args = parser.parse_args()


    experiments = []
    for seed in range(args.number_random_copies):
        for dataset_name in args.datasets:
            for sampling_mode in ['nlargest']:
                for training_fraction in [0.1, 0.25, 0.5]:
                    experiments.append({
                        'seed': seed,
                        'dataset_name': dataset_name,
                        'sampling_mode': sampling_mode,
                        'training_fraction': training_fraction,
                        'threads': 8
                    })

    print('There are', len(experiments), 'experiments in total')

    with open(args.experiments_file, 'w') as experiments_file:
        json.dump({'number_random_copies': args.number_random_copies,
                   'datasets': args.datasets,
                   'configurations': experiments}, experiments_file, sort_keys=True, indent=4)
