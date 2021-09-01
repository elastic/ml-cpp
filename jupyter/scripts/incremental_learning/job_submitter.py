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
import subprocess

from jinja2 import Environment, FileSystemLoader
from pathlib import Path

def init_task_spooler():
    cmd = ['tsp', '-S', '`nproc`']
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

def generate_job_file(config, cwd, verbose):

    job_name = '_'.join(map(str, [config['dataset_name'], config['seed'],
                                  config['threads'], config['transform_name'],
                                  config['transform_parameters'].get('fraction', ''),
                                  config['transform_parameters'].get('magnitude', '')]))
    job_file = '{}.job'.format(job_name)
    job_parameters = ['verbose='.format(verbose),
                      'dataset_name="{}"'.format(config['dataset_name']),
                      'threads={}'.format(config['threads']),
                      'seed={}'.format(config['seed']),
                      'transform_name="{}"'.format(config['transform_name']),
                      'transform_parameters="{}"'.format(config['transform_parameters'])]

    job = tm.render(job_name=job_name,
                    job_parameters=" ".join(job_parameters),
                    job_file=job_file, cwd=cwd)
    with open(job_file, 'wt') as fp:
        fp.write(job)
    os.chmod(job_file, 0o755)
    return job_file

def submit_to_task_spooler(threads, job_file_path):
    cmd = ['tsp', '-N', str(threads), str(job_file_path)]
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

if __name__ == '__main__':
    """
    This script reads the experiment configurations from the experiments.json file and
    creates individual job scripts using job.tpl template. These jobs are submitted to
    the task spooler for execution.
    """

    parser = argparse.ArgumentParser(description='Run a colletion of experiments defined by experiments.json')
    parser.add_argument('--verbose', default=False, action='store_true')
    args = parser.parse_args()

    cwd = os.getcwd()
    file_loader = FileSystemLoader(cwd)
    env = Environment(loader=file_loader)
    tm = env.get_template('job.tpl')
    init_task_spooler()
    with open('experiments.json') as fp:
        # TODO: validate schema of experiments.json
        experiments = json.load(fp)
    for config in experiments['configurations']:
        job_file = generate_job_file(config, cwd, args.verbose)
        job_file_path = Path(cwd)/job_file
        submit_to_task_spooler(config['threads'], job_file_path)
