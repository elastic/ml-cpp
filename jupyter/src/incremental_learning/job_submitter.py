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

import multiprocessing
import os
import subprocess

import jinja2
import pathlib2 as pathlib


def init_task_spooler() -> None:
    """Configures the task spooler to use maximum number of threads.
    """
    cmd = ['tsp', '-S', str(multiprocessing.cpu_count())]
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    process.communicate()


def submit_to_task_spooler(threads: int, job_file_path: pathlib.Path) -> None:
    """Submit the job file to the task spooler.

    Args:
        threads (int): Number of threads to use.
        job_file_path (pathlib.Path): Path to the job file.
    """
    cmd = ['tsp', '-N', str(threads), str(job_file_path)]
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    process.communicate()


def generate_job_file(config: dict, cwd: str, force_update: bool,
                      verbose: bool, comment: str, 
                      template: jinja2.Template) -> pathlib.Path:
    """Generate a job file instance from the template.

    Args:
        config (dict): Configuration used for template generation.
        cwd (str): Current working directory path.
        force_update (bool): Force update flag.
        verbose (bool): Verbosity flag.
        comment (str): Obligatory comment.
        template (jinja2.Template): The jinja2 template instance.

    Returns:
        pathlib.Path: Path to the generated job file.
    """

    job_name = '_'.join([str(v) for _, v in config.items()])
    job_file = '{}.job'.format(job_name)
    job_parameters = ['force_update={}'.format(force_update),
                      'verbose={}'.format(verbose)] + \
        ['{}={}'.format(k, v) for k, v in config.items()]

    job = template.render(job_name=job_name,
                          job_parameters=" ".join(job_parameters),
                          job_file=job_file, comment=comment, cwd=cwd)
    with open(job_file, 'wt') as fp:
        fp.write(job)
    os.chmod(job_file, 0o755)
    return pathlib.Path(cwd)/job_file
