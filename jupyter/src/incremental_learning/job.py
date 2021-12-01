# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.


import base64
import gzip
import json
import random
import string
import sys
import tempfile
import time
from pathlib import Path
from typing import Union

import libtmux
import numpy as np
import pandas
from deepmerge import always_merger
from IPython import display

from .config import configs_dir, dfa_path, logger
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

    def __init__(self, input: Union[None, str, tempfile._TemporaryFileWrapper] = None,
                 config: Union[None, str, tempfile._TemporaryFileWrapper] = None,
                 persist: Union[None, str, tempfile._TemporaryFileWrapper] = None,
                 restore: Union[None, str, tempfile._TemporaryFileWrapper] = None,
                 verbose: bool = True, run=None):
        """Initialize the class .

        Args:
            input (Union[str, tempfile._TemporaryFileWrapper]): input filename or temp file handler
            config (Union[str, tempfile._TemporaryFileWrapper]): config filename or temp file handler
            persist (Union[None, str, tempfile._TemporaryFileWrapper], optional): filename or temp file handler for model to persist to. Defaults to None.
            restore (Union[None, str, tempfile._TemporaryFileWrapper], optional): filename or temp file handler for model to restore from. Defaults to None.
            verbose (bool, optional): Verbosity. Defaults to True.
        """
        self.verbose = verbose

        self.input = input
        if is_temp(self.input):
            self.input_filename = self.input.name
        else:
            self.input_filename = self.input

        self.config = config
        if is_temp(self.config):
            self.config_filename = self.config.name
        else:
            self.config_filename = config

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
        self.run = run
        self.start_time = time.time()

        if self.input and self.config:
            self._set_dependent_variable_name()
            self._set_analysis_name()
            if self.is_classification():
                self._set_num_classes()
                self._configure_num_top_classes()
            self.initialized = True
        else:
            self.initialized = False

    def _set_dependent_variable_name(self):
        with open(self.config_filename) as fp:
            config = json.load(fp)
        self.dependent_variable = config['analysis']['parameters']['dependent_variable']

    def _set_analysis_name(self):
        with open(self.config_filename) as fp:
            config = json.load(fp)
        self.analysis_name = ""
        self.analysis_name = config['analysis']['name']

    def _set_num_classes(self):
        if self.is_classification():
            with open(self.config_filename) as fp:
                config = json.load(fp)
            self.num_classes = config['analysis']['parameters']['num_classes']

    def _configure_num_top_classes(self):
        with open(self.config_filename) as fc:
            config = json.load(fc)
        config['analysis']['parameters']['num_top_classes'] = self.num_classes
        with open(self.config_filename, 'wt') as fc:
            json.dump(config, fc)

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

    def wait_to_complete(self, clean=True) -> float:
        """Wait until the job is complete .

        Returns:
            float: Job execution time.
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
                    if not self.run:
                        print(err)
            # the line in main.cc where final output is produced
            success = sum(
                [line == 'Success' for line in self.pane.capture_pane()[-5:]]) == 1
            failure = sum(
                [line == 'Failure' for line in self.pane.capture_pane()[-5:]]) == 1
            if success or failure:
                break
            time.sleep(5.0)
        self.stop_time = time.time()

        if self.run and self.verbose:
            self.run.run_logger.info(err)

        if success:
            with open(self.output.name) as fp:
                self.results = json.load(fp)
            if is_temp(self.persist):
                with open(self.persist_filename) as fp:
                    self.model = "\n".join(fp.readlines()[-3:])
            if self.verbose:
                print('Job succeeded')
            if clean:
                self.clean()
        elif failure:
            self.results = {}
            if self.verbose:
                print('Job failed')
            if clean:
                self.clean()
            raise RuntimeError("Running data_frame_analyzer failed.")
        return self.stop_time - self.start_time

    def get_config(self) -> dict:
        with open(self.config.name) as fp:
            config = json.load(fp)
        return config

    def is_regression(self) -> bool:
        return self.analysis_name == 'regression'

    def is_classification(self) -> bool:
        return self.analysis_name == 'classification'

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

    def get_probabilities(self) -> Union[None, list]:
        """Return prediction probabilities for the class "true".

        Note: Only works for binary classification.

        Returns:
            Union[None, list]: prediction probabilities. 
        """
        if self.analysis_name == 'regression':
            raise ValueError(
                "Failed to obtain prediction probabilities for the regression job {}."
                .format(self.name))
        if self.num_classes != 2:
            raise NotImplementedError(
                "Only binary classification is implemented at the moment. Configured num_classes {}."
                .format(self.num_classes))
        probabilities = []
        for item in self.results:
            if 'row_results' in item:
                top_classes = item['row_results']['results']['ml']['top_classes']
                probabilities.append(
                    {x['class_name']: x['class_probability'] for x in top_classes})
        return probabilities

    def get_hyperparameters(self) -> dict:
        """Get all the hyperparameter values for the model.

        Returns:
            dict: hyperparameters
        """
        hyperparameters = {}
        for item in self.results:
            if 'model_metadata' in item:
                for hyperparameter in item['model_metadata']['hyperparameters']:
                    hyperparameters[hyperparameter['name']] = hyperparameter['value']
                hyperparameters['previous_train_num_rows'] = item['model_metadata']['train_properties']['num_train_rows']
                hyperparameters['previous_train_loss_gap'] = item['model_metadata']['train_properties']['loss_gap']
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
        result = []
        for item in self.results:
            if 'compressed_inference_model' in item:
                result.append(json.dumps(item))
        return "".join(result)

    def get_compressed_data_summarization(self) -> str:
        """
        Get compressed model definition json.
        """
        result = []
        for item in self.results:
            if 'compressed_data_summarization' in item:
                result.append(json.dumps(item))
        return "".join(result)

    def get_model_update_data(self) -> str:
        data_summarization = self.get_compressed_data_summarization()
        model_definition = self.get_compressed_inference_model()
        return data_summarization + "\0\n" + model_definition + "\0\n"

    def get_model_blob(self) -> str:
        return self.model

    def store(self, destination: Path) -> bool:
        """
        Store job with trained model into a file.

        Args:
            destination (Path): local path to where the job should be stored.

        Returns:
            bool: True if storing was successfull, False otherwise.
        """
        success = True
        with gzip.open(destination, 'wt') as fp:
            try:
                json.dump(obj=self, fp=fp, cls=JobJSONEncoder)
            except:
                ex = sys.exc_info()
                logger.error("Failed storing job {}: {}".format(self.name, ex))
                success = False
        return success

    @classmethod
    def as_job(cls, state: dict):
        if '__job__' in state:
            job = Job(input='', config='')
            job.input_filename = state['input_filename']
            job.config_filename = state['config_filename']
            job.verbose = state['verbose']
            job.dependent_variable = state['dependent_variable']
            job.analysis_name = state['analysis_name']
            job.persist_filename = state['persist_filename']
            job.restore_filename = state['restore_filename']
            job.name = state['name']
            job.model = state['model']
            if 'results' in state:
                job.results = state['results']
            job.initialized = True
        return job

    @classmethod
    def from_file(cls, source: Path):
        """
        Restore a Job object from file.

        Args:
            source (Path): local path to where the Job object should be restored from.

        Returns:
            Job: restored Job object.
        """
        if source.exists() == False or source.is_file() == False:
            logger.error('File to load Job from file. {} does not exist or is not a file'
                         .format(source))
            return None
        job = None
        with gzip.open(source, 'rt') as fp:
            state = json.load(fp=fp)
            job = cls.as_job(state)
        return job

    def __eq__(self, other):
        return self.input_filename == other.input_filename \
            and self.config_filename == other.config_filename \
            and self.verbose == other.verbose \
            and self.dependent_variable == other.dependent_variable \
            and self.analysis_name == other.analysis_name \
            and self.persist_filename == other.persist_filename \
            and self.restore_filename == other.restore_filename \
            and self.name == other.name \
            and self.model == other.model \
            and (hasattr(self, 'results') == hasattr(other, 'results') and (hasattr(self, 'results') == False or self.results == other.results))


class JobJSONEncoder(json.JSONEncoder):
    """
    JSON serialization logic for the Job class.
    """

    def default(self, obj: Job):
        if isinstance(obj, Job):
            state = {
                'input_filename': obj.input_filename,
                'config_filename': obj.config_filename,
                'verbose': obj.verbose,
                'dependent_variable': obj.dependent_variable,
                'analysis_name': obj.analysis_name,
                'persist_filename': obj.persist_filename,
                'restore_filename': obj.restore_filename,
                'name': obj.name,
                'model': obj.model,
                '__job__': True
            }
            if hasattr(obj, 'results'):
                state['results'] = obj.results
            return state
        return json.JSONEncoder.default(self, obj)

def update_config(config, run=None) -> dict:
    if run:
        if 'threads' in run.config.keys():
            config['threads'] = run.config['threads']
        if 'seed' in run.config.keys():
            config['analysis']['parameters']['seed'] = run.config['seed']
        if 'analysis' in run.config.keys() and 'parameters' in run.config['analysis']:
            always_merger.merge(config['analysis']['parameters'], run.config['analysis']['parameters'])
    return config


def run_job(input, config, persist=None, restore=None, verbose=True, run=None) -> Job:
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
              restore=restore, verbose=verbose, run=run)
    job_suffix = ''.join(random.choices(
        string.ascii_lowercase, k=5))+job.config_filename
    job_name = 'job_{}'.format(job_suffix)

    cmd = [str(dfa_path),
           "--input", job.input_filename,
           "--config", job.config_filename,
           "--output", job.output.name,
           "--validElasticLicenseKeyConfirmed", "true"]

    if job.persist:
        cmd += ["--persist", job.persist_filename]
    if job.restore:
        cmd += ['--restore', job.restore_filename]

    cmd = ' '.join(cmd)
    cmd += '; if [ $? -eq 0 ]; then echo "Success"; else echo "Failure";  fi;'

    if run and verbose:
        run.run_logger.info(cmd)

    session = server.new_session(job_name)
    window = session.new_window(attach=False)
    pane = window.split_window(attach=False)

    if run is None and verbose:
        print("session: {}\tcommand:\n{}".format(
            session.get('session_name'), cmd))
    pane.send_keys(cmd)
    job.name = job_name
    job.pane = pane
    return job


def train(dataset_name: str, dataset: pandas.DataFrame, verbose: bool = True, run=None) -> Job:
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
    config = update_config(config, run=run)

    config_file = tempfile.NamedTemporaryFile(mode='wt')
    json.dump(config, config_file)
    config_file.file.close()

    model_file = tempfile.NamedTemporaryFile(mode='wt')

    job = run_job(input=data_file, config=config_file,
                  persist=model_file, verbose=verbose, run=run)
    return job


def evaluate(dataset_name: str, dataset: pandas.DataFrame, original_job: Job, verbose: bool = True, run=None) -> Job:
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
    config = update_config(config, run=run)
    config['analysis']['parameters']['task'] = 'predict'

    fconfig = tempfile.NamedTemporaryFile(mode='wt')
    json.dump(config, fconfig)
    fconfig.file.close()
    fmodel = tempfile.NamedTemporaryFile(mode='wt')
    fmodel.write(original_job.get_model_update_data())
    fmodel.file.close()
    job = run_job(input=fdata, config=fconfig,
                  restore=fmodel, verbose=verbose, run=run)
    return job


def update(dataset_name: str,
           dataset: pandas.DataFrame,
           original_job: Job,
           force: bool = False,
           verbose: bool = True,
           run = None) -> Job:
    """Train a new model incrementally using the model and hyperparameters from the original job.

    Args:
        dataset_name (str): dataset name (to get configuration)
        dataset (pandas.DataFrame): new dataset
        original_job (Job): Job object of the original training
        force (bool): Force accept the update (default: False)
        verbose (bool): Verbosity flag (default: True)
        run: The run context (default: None)

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
    config = update_config(config, run=run)

    if force:
        config['analysis']['parameters']['force_accept_incremental_training'] = True

    for name, value in original_job.get_hyperparameters().items():
        if name not in ['retrained_tree_eta', 'tree_topology_change_penalty', 'soft_tree_depth_limit', 'gamma', 'alpha']:
            config['analysis']['parameters'][name] = value
        if name in ['gamma', 'alpha']:
            min, max = (value*0.5, value)
            config['analysis']['parameters'][name] = [min, max]
            logger.debug("{name}: past value {value}, new range [{min},{max}]".format(
                value=value,min=min, max=max, name=name))
        if name in ['soft_tree_depth_limit']:
            min, max = (value, value*1.5)
            config['analysis']['parameters'][name] = [min, max]
            logger.debug("{name}: past value {value}, new range [{min},{max}]".format(
                value=value,min=min, max=max, name=name))
        if name == 'retrained_tree_eta':
            logger.debug("retrained_tree_eta value is {}".format(value))
        if name == 'retrained_tree_eta':
            min, max = (value/2, value*2)
            min, max = tuple(np.clip([min, max], 0.1, 0.6))
            config['analysis']['parameters'][name] = [min, max]
            logger.debug("retrained_tree_eta: past value {value}, new range [{min},{max}] clipped to {clipped}".format(
                value=value,min=min, max=max, clipped=config['analysis']['parameters'][name]))
    config['analysis']['parameters']['task'] = 'update'
    fconfig = tempfile.NamedTemporaryFile(mode='wt')
    json.dump(config, fconfig)
    fconfig.file.close()
    fmodel = tempfile.NamedTemporaryFile(mode='wt')
    fmodel.write(original_job.get_model_update_data())
    fmodel.file.close()
    job = run_job(input=fdata, config=fconfig,
                  restore=fmodel, verbose=verbose, run=run)
    return job
