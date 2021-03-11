import numpy as np
import scipy.spatial
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
import base64
import gzip
import heapq
from operator import itemgetter
from .trees import Tree

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


def decompress(compressed_str: str):
    base64_bytes = compressed_str
    gzip_bytes = base64.b64decode(base64_bytes)
    message_bytes = gzip.decompress(gzip_bytes)
    message = message_bytes.decode('utf8')
    return message


class Job:
    def __init__(self, input: Union[str, tempfile._TemporaryFileWrapper],
                 config: Union[str, tempfile._TemporaryFileWrapper],
                 persist: Union[None, str, tempfile._TemporaryFileWrapper] = None,
                 restore: Union[None, str, tempfile._TemporaryFileWrapper] = None,
                 verbose: bool = True):
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

    def wait_to_complete(self) -> bool:
        while True:
            err = "\n".join(self.pane.capture_pane())
            with open(self.output.name) as f:
                out = f.readlines()[-10:]
                out = [l[:80] for l in out]
            # out =  !grep "phase_progress" $output.name | tail -n10
            out = "\n".join(out)
            if self.verbose:
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
            if is_temp(self.persist):
                with open(self.persist_filename) as fp:
                    self.model = "\n".join(fp.readlines()[-3:])
            if self.verbose:
                print('Job succeeded')
            self.cleanup()
            return True
        elif failure:
            self.results = {}
            if self.verbose:
                print('Job failed')
            self.cleanup()
            return False

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
        for item in self.results:
            if 'compressed_inference_model' in item:
                base64_bytes = item['compressed_inference_model']['definition']
                gzip_bytes = base64.b64decode(base64_bytes)
                message_bytes = gzip.decompress(gzip_bytes)
                message = message_bytes.decode('utf8')
                definition = json.loads(message)
                return definition
        return {}

    def get_model_blob(self) -> str:
        return self.model


def run_job(input, config, persist=None, restore=None, verbose=True) -> Job:
    job = Job(input=input, config=config, persist=persist,
              restore=restore, verbose=verbose)
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
    dataset.to_csv(data_file, index=False)
    data_file.file.close()

    with open('configs/{}.json'.format(dataset_name)) as fc:
        config = json.load(fc)
    config['rows'] = dataset.shape[0]
    config_file = tempfile.NamedTemporaryFile(mode='wt')
    json.dump(config, config_file)
    config_file.file.close()

    model_file = tempfile.NamedTemporaryFile(mode='wt')

    job = run_job(input=data_file, config=config_file,
                  persist=model_file, verbose=verbose)
    return job


def evaluate(dataset_name: str, dataset: pandas.DataFrame, model: str, verbose: bool = True) -> Job:
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
    job = run_job(input=fdata, config=fconfig, restore=fmodel, verbose=verbose)
    return job


def get_kdtree_samples(X: np.array, nsamples: int) -> np.array:
    def traverse_kdtree(node, max_level, split_dim, split_value):
        if node is not None and node.level <= max_level:
            if node.split_dim >= 0:
                split_dim.append(node.split_dim)
                split_value.append(node.split)
                split_dim, split_value = traverse_kdtree(
                    node.lesser, max_level, split_dim, split_value)
                split_dim, split_value = traverse_kdtree(
                    node.greater, max_level, split_dim, split_value)
        return split_dim, split_value

    kdtree = scipy.spatial.cKDTree(X)
    max_level = np.ceil(np.log2(nsamples))
    split_dim, split_value = traverse_kdtree(
        node=kdtree.tree, max_level=max_level, split_dim=[], split_value=[])
    centers = np.where(X[:, split_dim] == split_value)[0]
    return X[centers[:nsamples], :]


def get_covertree_samples(X: np.array, nsamples: int) -> np.array:
    from . import covertree
    leafsize = 10
    ctree = covertree.CoverTree(
        data=X, distance=scipy.spatial.distance.euclidean, leafsize=leafsize)
    leaf_idx = int(leafsize*nsamples/X.shape[0])
    num_centers = 0
    queue = [ctree.root]
    centers = []
    while num_centers < nsamples and len(queue) > 0:
        node = queue.pop(0)
        if isinstance(node, covertree.CoverTree._InnerNode):
            centers.append(node.ctr_idx)
            for child in node.children:
                queue.append(child)
        elif isinstance(node, covertree.CoverTree._LeafNode):
            centers += node.idx[:leaf_idx]
        num_centers = pandas.unique(centers).shape[0]
    centers = pandas.unique(centers)
    # print("Num samples {}".format(num_centers))
    return X[list(centers)[:nsamples], :]


def summarize(dataset_name: str,
              dataset: pandas.DataFrame,
              size: Union[int, float],
              model: str,
              method: str = 'random',
              verbose: bool = True) -> pandas.DataFrame:
    total_size = dataset.shape[0]
    if isinstance(size, float):
        sample_size = int(total_size*size)
    else:
        sample_size = size
    if method == 'random':
        return dataset.sample(n=sample_size)
    if method == 'diversity':
        import diversipy
        summarization = diversipy.subset.psa_select(
            dataset.to_numpy(), sample_size)
        summarization_df = pandas.DataFrame(
            data=summarization, columns=dataset.columns)
        return summarization_df
    if method == 'kdtree':
        summarization = get_kdtree_samples(dataset.to_numpy(), sample_size)
        summarization_df = pandas.DataFrame(
            data=summarization, columns=dataset.columns)
        return summarization_df
    if method == 'cover-tree':
        summarization = get_covertree_samples(dataset.to_numpy(), sample_size)
        summarization_df = pandas.DataFrame(
            data=summarization, columns=dataset.columns)
        return summarization_df


class DataStream():
    def __init__(self, dataset: pandas.DataFrame, tree: Tree):
        self.dataset = dataset
        self.current_index = 0
        self.tree = tree

    def length(self) -> int:
        return self.dataset.shape[0]-self.current_index

    def weight(self) -> float:
        node = self.tree.nodes[self.tree.get_leaf_id(
            self.dataset.loc[self.current_index])]
        num_samples = node.number_samples
        return float(num_samples)
        # return 1.0

    def next(self):
        self.current_index += 1

    def current(self):
        return self.dataset.loc[self.current_index]

    def empty(self) -> bool:
        return self.current_index == self.dataset.shape[0]

    def columns(self):
        return self.dataset.columns


def reservoir_sample_with_jumps(S: DataStream, k: int):
    H = []  # min priority queue
    while not S.empty() and len(H) < k:
        r = random.random()**(1.0/S.weight())
        heapq.heappush(H, (r, S.current()))
        S.next()
    Hmin = H[0][0]
    X = np.log(random.random())/np.log(Hmin)
    while not S.empty():
        X -= S.weight()
        if X <= 0:
            t = Hmin**S.weight()
            r = random.uniform(t, 1)**(1.0/S.weight())
            heapq.heappop(H)
            heapq.heappush(H, (r, S.current()))
            Hmin = H[0][0]
            X = np.log(random.random())/np.log(Hmin)
        S.next()
    return pandas.DataFrame(data=map(itemgetter(1), H), columns=S.columns())
