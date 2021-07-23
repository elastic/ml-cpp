# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.

import base64
import gzip
import json
import random
import string
import tempfile
import time
from typing import Union

import libtmux
import numpy as np
import pandas
from IPython import display

from .config import configs_dir, datasets_dir, dfa_path
from .misc import isnotebook

server = libtmux.Server()


def is_temp(obj: object):
    return isinstance(obj, tempfile._TemporaryFileWrapper)


def decompress(compressed_str: str):
    base64_bytes = compressed_str
    gzip_bytes = base64.b64decode(base64_bytes)
    message_bytes = gzip.decompress(gzip_bytes)
    message = message_bytes.decode('utf8')
    return message


class Job:
    """Job class .
    """

    def __init__(self, input: Union[str, tempfile._TemporaryFileWrapper],
                 config: Union[str, tempfile._TemporaryFileWrapper],
                 persist: Union[None, str, tempfile._TemporaryFileWrapper] = None,
                 restore: Union[None, str, tempfile._TemporaryFileWrapper] = None,
                 verbose: bool = True):
        """Initialize the class .

        Args:
            input (Union[str, tempfile._TemporaryFileWrapper]): input filename or temp file handler
            config (Union[str, tempfile._TemporaryFileWrapper]): config filename or temp file handler
            persist (Union[None, str, tempfile._TemporaryFileWrapper], optional): filename or temp file handler for model to persist to. Defaults to None.
            restore (Union[None, str, tempfile._TemporaryFileWrapper], optional): filename or temp file handler for model to restore from. Defaults to None.
            verbose (bool, optional): Verbosity. Defaults to True.
        """
        self.input = input
        self.config = config
        self.verbose = verbose

        if is_temp(self.input):
            self.input_filename = self.input.name
        else:
            self.input_filename = self.input
        if is_temp(self.config):
            self.config_filename = self.config.name
        else:
            self.config_filename = config
        self._set_dependent_variable_name()

        self.persist = persist
        if is_temp(self.persist):
            self.persist_filename = self.persist.name
        else:
            self.persist_filename = self.persist

        self.restore = restore
        if is_temp(self.restore):
            self.restore_filename = self.restore.name
        else:
            self.restore_filename = self.restore

        self.name = ''
        self.pane = None
        self.output = tempfile.NamedTemporaryFile(mode='wt')
        self.model = ''

    def _set_dependent_variable_name(self):
        with open(self.config_filename) as fp:
            config = json.load(fp)
        self.dependent_variable = config['analysis']['parameters']['dependent_variable']

    def clean(self):
        if is_temp(self.output):
            self.output.close()
        if is_temp(self.persist):
            self.persist.close()
        if is_temp(self.restore):
            self.restore.close()
        if is_temp(self.config):
            self.config.close()
        if is_temp(self.input):
            self.input.close()
        if self.name:
            server.kill_session(target_session=self.name)

    def wait_to_complete(self) -> bool:
        """Wait until the job is complete .

        Returns:
            bool: True if job run successfully, False otherwise.
        """
        while True:
            err = "\n".join(self.pane.capture_pane())
            with open(self.output.name) as f:
                out = f.readlines()[-10:]
                out = [l[:80] for l in out]
            # out =  !grep "phase_progress" $output.name | tail -n10
            out = "\n".join(out)
            if self.verbose:
                if isnotebook():
                    display.clear_output(wait=True)
                    display.display_html(display.HTML("""<table><tr>
                                                                <td width="50%" style="text-align:center;"><b>stderr</b></td>
                                                                <td width="50%" style="text-align:center;"><b>output</b></td>
                                                                </tr>
                                                                <tr>
                                                                <td width="50%" style="text-align:left;"><pre>{}</pre></td>
                                                                <td width="50%" style="text-align:left;"><pre>{}</pre></td>
                                                                </tr>
                                                        </table>"""
                                                    .format(err, out)))
                else:
                    print(err)
            # the line in main.cc where final output is produced
            success = sum(
                [line == 'Success' for line in self.pane.capture_pane()[-5:]]) == 1
            failure = sum(
                [line == 'Failure' for line in self.pane.capture_pane()[-5:]]) == 1
            if success or failure:
                break
            time.sleep(5.0)

        if success:
            with open(self.output.name) as fp:
                self.results = json.load(fp)
            if is_temp(self.persist):
                with open(self.persist_filename) as fp:
                    self.model = "\n".join(fp.readlines()[-3:])
            if self.verbose:
                print('Job succeeded')
            self.clean()
            return True
        elif failure:
            self.results = {}
            if self.verbose:
                print('Job failed')
            self.clean()
            return False

    def get_predictions(self) -> np.array:
        """Returns a numpy array of the predicted values for the model

        Returns:
            np.array: predictions
        """
        predictions = []
        for item in self.results:
            if 'row_results' in item:
                predictions.append(item['row_results']['results']
                                   ['ml']['{}_prediction'.format(self.dependent_variable)])
        return np.array(predictions)

    def get_hyperparameters(self) -> dict:
        """Get all the hyperparameter values for the model.

        Returns:
            dict: hyperparameters
        """
        hyperparameters = {}
        for item in self.results:
            if 'model_metadata' in item:
                for hyperparameter in item['model_metadata']['hyperparameters']:
                    hyperparameters[hyperparameter['name']
                                    ] = hyperparameter['value']
        return hyperparameters

    def get_data_summarization_num_rows(self) -> int:
        """Get the number of examples used in data summarization.

        Returns:
            int: number of examples
        """
        num_rows = 0
        for item in self.results:
            if 'model_metadata' in item:
                if 'data_summarization' in item['model_metadata']:
                    num_rows = item['model_metadata']['data_summarization']['num_rows']
        return num_rows

    def get_model_definition(self) -> dict:
        """
        Get model definition json.

        Usage:
        definition = get_model_definition(results)
        trained_models = definition['trained_model']['ensemble']['trained_models']
        forest = Forest(trained_models)
        """
        for item in self.results:
            if 'compressed_inference_model' in item:
                base64_bytes = item['compressed_inference_model']['definition']
                gzip_bytes = base64.b64decode(base64_bytes)
                message_bytes = gzip.decompress(gzip_bytes)
                message = message_bytes.decode('utf8')
                definition = json.loads(message)
                return definition
        return {}

    def get_compressed_inference_model(self) -> str:
        """
        Get compressed model definition json.
        """
        for item in self.results:
            if 'compressed_inference_model' in item:
                return json.dumps(item)
        return ""

    def get_compressed_data_summarization(self) -> str:
        """
        Get compressed model definition json.
        """
        for item in self.results:
            if 'compressed_data_summarization' in item:
                return json.dumps(item)
        return ""

    def get_model_update_data(self) -> str:
        data_summarization = self.get_compressed_data_summarization()
        model_definition = self.get_compressed_inference_model()
        return data_summarization + "\0\n" + model_definition + "\0\n"

    def get_model_blob(self) -> str:
        return self.model


def run_job(input, config, persist=None, restore=None, verbose=True) -> Job:
    """Run a DFA job.

    Args:
        input (str or temp file handler ): Dataset
        config (str or temp file handler): Configuration
        persist (str or temp file handler, optional): Persist model to file. Defaults to None.
        restore (str or temp file handler, optional): Restore model from file. Defaults to None.
        verbose (bool, optional): Verbosity. Defaults to True.

    Returns:
        Job: job object with job process started asynchronously.
    """
    job = Job(input=input, config=config, persist=persist,
              restore=restore, verbose=verbose)
    job_suffix = ''.join(random.choices(string.ascii_lowercase, k=5))
    job_name = 'job_{}'.format(job_suffix)

    cmd = [str(dfa_path),
           "--input", job.input_filename,
           "--config", job.config_filename,
           "--output", job.output.name]

    if job.persist:
        cmd += ["--persist", job.persist_filename]
    if job.restore:
        cmd += ['--restore', job.restore_filename]

    cmd = ' '.join(cmd)
    cmd += '; if [ $? -eq 0 ]; then echo "Success"; else echo "Failure";  fi;'

    session = server.new_session(job_name)
    window = session.new_window(attach=False)
    pane = window.split_window(attach=False)

    if verbose:
        print("session: {}\tcommand:\n{}".format(
            session.get('session_name'), cmd))
    pane.send_keys(cmd)
    job.name = job_name
    job.pane = pane
    return job


def train(dataset_name: str, dataset: pandas.DataFrame, verbose: bool = True) -> Job:
    """Train a model on the dataset .

    Args:
        dataset_name (str): [description]
        dataset (pandas.DataFrame): [description]

    Returns:
        Job: [description]
    """
    data_file = tempfile.NamedTemporaryFile(mode='wt')
    dataset.to_csv(data_file, index=False, na_rep='\0')
    data_file.file.close()

    with open(configs_dir / '{}.json'.format(dataset_name)) as fc:
        config = json.load(fc)
    config['rows'] = dataset.shape[0]
    config_file = tempfile.NamedTemporaryFile(mode='wt')
    json.dump(config, config_file)
    config_file.file.close()

    model_file = tempfile.NamedTemporaryFile(mode='wt')

    job = run_job(input=data_file, config=config_file,
                  persist=model_file, verbose=verbose)
    return job


def evaluate(dataset_name: str, dataset: pandas.DataFrame, original_job: Job, verbose: bool = True) -> Job:
    """Evaluate the model on a given dataset .

    Args:
        dataset_name (str): dataset name (to get configuration)
        dataset (pandas.DataFrame): new dataset
        original_job (Job): Job object of the original training
        verbose (bool): Verbosity flag (defalt: True)

    Returns:
        Job: new Job object
    """
    fdata = tempfile.NamedTemporaryFile(mode='wt')
    dataset.to_csv(fdata, index=False, na_rep=0)
    fdata.file.close()
    with open(configs_dir / '{}.json'.format(dataset_name)) as fc:
        config = json.load(fc)
    config['rows'] = dataset.shape[0] + \
        original_job.get_data_summarization_num_rows()
    config['analysis']['parameters']['task'] = 'predict'
    fconfig = tempfile.NamedTemporaryFile(mode='wt')
    json.dump(config, fconfig)
    fconfig.file.close()
    fmodel = tempfile.NamedTemporaryFile(mode='wt')
    fmodel.write(original_job.get_model_update_data())
    fmodel.file.close()
    job = run_job(input=fdata, config=fconfig, restore=fmodel, verbose=verbose)
    return job


def update(dataset_name: str, dataset: pandas.DataFrame, original_job: Job, verbose: bool = True) -> Job:
    """Train a new model incrementally using the model and hyperparameters from the original job.

    Args:
        dataset_name (str): dataset name (to get configuration)
        dataset (pandas.DataFrame): new dataset
        original_job (Job): Job object of the original training
        verbose (bool): Verbosity flag (defalt: True)

    Returns:
        Job: new Job object
    """
    fdata = tempfile.NamedTemporaryFile(mode='wt')
    dataset.to_csv(fdata, index=False, na_rep=0)
    fdata.file.close()
    with open(configs_dir / '{}.json'.format(dataset_name)) as fc:
        config = json.load(fc)
    config['rows'] = dataset.shape[0] + \
        original_job.get_data_summarization_num_rows()
    for name, value in original_job.get_hyperparameters().items():
        if name not in ['retrained_tree_eta', 'tree_topology_change_penalty']:
            config['analysis']['parameters'][name] = value
    config['analysis']['parameters']['task'] = 'update'
    fconfig = tempfile.NamedTemporaryFile(mode='wt')
    json.dump(config, fconfig)
    fconfig.file.close()
    fmodel = tempfile.NamedTemporaryFile(mode='wt')
    fmodel.write(original_job.get_model_update_data())
    fmodel.file.close()
    job = run_job(input=fdata, config=fconfig, restore=fmodel, verbose=verbose)
    return job
