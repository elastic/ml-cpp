import numpy as np
import pandas
import tempfile
import json
import random
import string
import platform
import os
import libtmux
from IPython import display
import time
from typing import Union

# I assume, your host OS is not CentOS
cloud = (platform.system() == 'Linux') and (platform.dist()[0] == 'centos')
if cloud:
    mlcpp_root = '/ml-cpp'
else:
    system = '{}-{}'.format(platform.system().lower(),
                            platform.machine().lower())
    mlcpp_root = os.environ['CPP_SRC_HOME'] + \
        '/build/distribution/platform/{}'.format(system)
dfa_path = mlcpp_root+'/bin/data_frame_analyzer'

server = libtmux.Server()


def is_temp(obj: object):
    return isinstance(obj, tempfile._TemporaryFileWrapper)


class Job:
    def __init__(self, input: Union[str, tempfile._TemporaryFileWrapper],
                 config: Union[str, tempfile._TemporaryFileWrapper],
                 persist: Union[None, str, tempfile._TemporaryFileWrapper] = None,
                 restore: Union[None, str, tempfile._TemporaryFileWrapper] = None):
        self.input = input
        self.config = config

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

    def cleanup(self):
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

    def wait_job_complete(self):
        while True:
            display.clear_output(wait=True)
            err = "\n".join(self.pane.capture_pane())
            with open(self.output.name) as f:
                out = f.readlines()[-10:]
                out = [l[:80] for l in out]
            # out =  !grep "phase_progress" $output.name | tail -n10
            out = "\n".join(out)
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
            # the line in main.cc where final output is produced
            success = sum(
                ['Main.cc@246' in line for line in self.pane.capture_pane()[-5:]]) == 1
            failure = sum(
                ['FATAL' in line for line in self.pane.capture_pane()[-5:]]) == 1
            if success or failure:
                break
            time.sleep(5.0)

        if success:
            with open(self.output.name) as fp:
                self.results = json.load(fp)
            print("Getting model from {} {}".format(is_temp(self.persist), self.persist_filename))
            if is_temp(self.persist):
                with open(self.persist_filename) as fp:
                    self.model = "\n".join(fp.readlines()[-3:])
            print('Job succeeded')
        elif failure:
            self.results = {}
            print('Job failed')
        self.cleanup()


    def get_predictions(self) -> np.array:
        predictions = []
        for item in self.results:
            if 'row_results' in item:
                predictions.append(item['row_results']['results']
                                ['ml']['{}_prediction'.format(self.dependent_variable)])
        return np.array(predictions)


    def get_hyperparameters(self) -> dict:
        hyperparameters = {}
        for item in self.results:
            if 'model_metadata' in item:
                for hyperparameter in item['model_metadata']['hyperparameters']:
                    hyperparameters[hyperparameter['name']
                                    ] = hyperparameter['value']
        return hyperparameters


    def get_model_definition(self) -> dict:
        """Usage:
        definition = get_model_definition(results)
        trained_models = definition['trained_model']['ensemble']['trained_models']
        forest = Forest(trained_models)
        """
        import base64
        import gzip
        for item in self.results:
            if 'compressed_inference_model' in item:
                base64_bytes = item['compressed_inference_model']['definition']
                gzip_bytes = base64.b64decode(base64_bytes)
                message_bytes = gzip.decompress(gzip_bytes)
                message = message_bytes.decode('utf8')
                definition = json.loads(message)
                return definition
        return {}

    def get_model_blob(self) ->str:
        return self.model


def run_job(input, config, persist=None, restore=None)->Job:
    job = Job(input=input, config=config, persist=persist, restore=restore)
    job_suffix = ''.join(random.choices(string.ascii_lowercase, k=5))
    job_name = 'job_{}'.format(job_suffix)

    cmd = [dfa_path,
           "--input", job.input_filename,
           "--config", job.config_filename,
           "--output", job.output.name]

    if job.persist:
        cmd += ["--persist", job.persist_filename]
    if job.restore:
        cmd += ['--restore', job.restore_filename]

    cmd = ' '.join(cmd)

    session = server.new_session(job_name)
    window = session.new_window(attach=False)
    pane = window.split_window(attach=False)

    print("session: {}\tcommand:\n{}".format(session.get('session_name'), cmd))
    pane.send_keys(cmd)
    job.name = job_name
    job.pane = pane
    return job


def train(dataset_name: str, dataset: pandas.DataFrame) -> Job:
    """Train a model on the dataset .

    Args:
        dataset_name (str): [description]
        dataset (pandas.DataFrame): [description]

    Returns:
        Job: [description]
    """
    data_file = tempfile.NamedTemporaryFile(mode='wt')
    dataset.to_csv(data_file, index=False)
    data_file.file.close()

    with open('configs/{}.json'.format(dataset_name)) as fc:
        config = json.load(fc)
    config['rows'] = dataset.shape[0]
    config_file = tempfile.NamedTemporaryFile(mode='wt')
    json.dump(config, config_file)
    config_file.file.close()
    
    model_file = tempfile.NamedTemporaryFile(mode='wt')
    
    job = run_job(input=data_file, config=config_file, persist=model_file)
    return job


def evaluate(dataset_name: str, dataset: pandas.DataFrame, model: str) -> Job:
    """Evaluate the model on a given dataset .

    Args:
        dataset_name (str): [description]
        dataset (pandas.DataFrame): [description]
        model (str): [description]

    Returns:
        Job: [description]
    """
    fdata = tempfile.NamedTemporaryFile(mode='wt')
    dataset.to_csv(fdata, index=False)
    fdata.file.close()
    with open('configs/{}.json'.format(dataset_name)) as fc:
        config = json.load(fc)
    config['rows'] = dataset.shape[0]
    fconfig = tempfile.NamedTemporaryFile(mode='wt')
    json.dump(config, fconfig)
    fconfig.file.close()
    fmodel = tempfile.NamedTemporaryFile(mode='wt')
    fmodel.write(model)
    fmodel.file.close()
    job = run_job(input=fdata, config=fconfig, restore=fmodel)
    return job


def summarize(dataset_name: str, dataset: pandas.DataFrame, nrows: int,  model: str) -> pandas.DataFrame:
    # TODO: implement

    return dataset.sample(nrows)
