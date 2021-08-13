import json
import subprocess
import os

if __name__ == '__main__':
    with open('experiments.json') as fp:
        # TODO: validate schema of experiments.json
        experiments = json.load(fp)
    for config in experiments['configurations']:
        cmd = ['python', 'experiment_driver.py', '--force',
                                    'with', 'dataset_name="{}"'.format(
                                        config['dataset_name']),
                                    'seed={}'.format(config['seed']),
                                    'transform_name="{}"'.format(
                                        config['transform_name']),
                                   'transform_parameters="{}"'.format(
                                       config['transform_parameters'])]
        print (' '.join(cmd))
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)
