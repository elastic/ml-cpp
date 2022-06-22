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
import json
import os

from incremental_learning.job_submitter import (generate_job_file,
                                                init_task_spooler,
                                                submit_to_task_spooler)
from jinja2 import Environment, FileSystemLoader

if __name__ == '__main__':
    """
    This script reads the experiment configurations from the experiments.json file and
    creates individual job scripts using job.tpl template. These jobs are submitted to
    the task spooler for execution.
    """

    parser = argparse.ArgumentParser(
        description='Run a collection of experiments defined by experiments.json')
    parser.add_argument('--verbose',
                        default=False,
                        action='store_true',
                        help='Verbose logging')
    parser.add_argument('--force_update',
                        default=False,
                        action='store_true',
                        help='Force accept the result of incremental training')
    parser.add_argument('-c', '--comment',
                        required=True,
                        help='A user defined comment for this set of runs')
    args = parser.parse_args()

    cwd = os.getcwd()
    file_loader = FileSystemLoader(cwd)
    env = Environment(loader=file_loader)
    tm = env.get_template('job.tpl')
    init_task_spooler()
    with open('experiments.json') as fp:
        experiments = json.load(fp)
    for config in experiments['configurations']:
        job_file_path = generate_job_file(config=config,
                                     cwd=cwd,
                                     force_update=args.force_update,
                                     verbose=args.verbose,
                                     comment=args.comment,
                                     template=tm)
        submit_to_task_spooler(config['threads'], job_file_path)
