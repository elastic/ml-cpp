#!/usr/bin/env python3
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
#  Run unit test for a given module in parallel.
#
import os
import sys
import subprocess
from multiprocessing import Pool, cpu_count
from datetime import datetime


def update_progress(completed, total, status):
    bar_len = 30
    completed_len = int(round(bar_len * completed / float(total)))
    completed_percents = round(100.0 * completed / float(total), 1)
    bar_filled = '=' * completed_len
    status = status[:80] # truncate test_suite name
    sys.stdout.write('\r[{bar:-<{bar_len}}] {completed}% ...{status: <80}'.format(bar=bar_filled, bar_len=bar_len, completed=completed_percents, status=status))
    sys.stdout.flush()


def get_test_cases(driver):
    result = []
    content = subprocess.run([driver, '--list_content'], stderr=subprocess.PIPE).stderr.decode('ascii').strip()
    for line in content.split("\n"):
        if not line.startswith('    '):
            test_suite = line.strip(' *')
        else:
            test_case = line.strip(' *')
            result.append("{}/{}".format(test_suite, test_case))
    return result


def worker(test_suite):
    proc = subprocess.Popen(['./ml_test', '--run_test={}'.format(test_suite)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    o, e = proc.communicate()
    if proc.returncode != 0:
        print(o.decode('ascii'))
    return test_suite


def run(driver, test_cases, num_cpus):
    start_time = datetime.now()
    total = len(test_cases)
    update_progress(0, total, '')
    with Pool(num_cpus) as pp:
        for i, test_case in enumerate(pp.imap_unordered(worker, test_cases)):
            update_progress(i + 1, total, test_case)
    elapsed_time = datetime.now() - start_time
    print("\nAll tests completed in {time:3f}s".format(time=elapsed_time.total_seconds()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run all unit test for a given module in parallel.")
    parser.add_argument('module',  help='module to test')
    parser.add_argument('--cpu', type=int, dest='num_cpus', default=cpu_count(),
                        help='number of CPUs to use, default: {}'.format(cpu_count()))
    args = parser.parse_args()

    mlcpp_dir = os.environ['CPP_SRC_HOME']
    module = args.module
    print("Running test for module {} on {} cpus".format(module, args.num_cpus))
    workdir = '{}/lib/{}/unittest'.format(mlcpp_dir, module)
    os.chdir(workdir)
    driver = './ml_test'.format(workdir, module)
    test_cases = get_test_cases(driver)
    run(driver, test_cases, args.num_cpus)
