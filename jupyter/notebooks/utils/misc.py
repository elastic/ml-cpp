# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.

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
from .trees import Tree, Forest

# I assume, your host OS is not CentOS
cloud = (platform.system() == 'Linux') and (platform.dist()[0] == 'centos')
if cloud:
    mlcpp_root = '/ml-cpp'
else:
    system = '{}-{}'.format(platform.system().lower(),
                            platform.machine().lower())
    mlcpp_root = os.environ['CPP_SRC_HOME'] + \
        '/build/distribution/platform/{}'.format(system)

# Adjust the path for MacOS
mac = (platform.system() == 'Darwin')
if mac:
    dfa_path = mlcpp_root+'/controller.app/Contents/MacOS/data_frame_analyzer'
else:
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
                print(err)
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

    with open('../configs/{}.json'.format(dataset_name)) as fc:
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
    with open('../configs/{}.json'.format(dataset_name)) as fc:
        config = json.load(fc)
    config['rows'] = dataset.shape[0]
    config['analysis']['parameters']['task'] = 'predict'
    fconfig = tempfile.NamedTemporaryFile(mode='wt')
    json.dump(config, fconfig)
    fconfig.file.close()
    fmodel = tempfile.NamedTemporaryFile(mode='wt')
    fmodel.write(model)
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
    dataset.to_csv(fdata, index=False)
    fdata.file.close()
    with open('../configs/{}.json'.format(dataset_name)) as fc:
        config = json.load(fc)
    config['rows'] = dataset.shape[0]
    for name, value  in original_job.get_hyperparameters().items():
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
              model_definition,
              method: str = 'random',
              verbose: bool = True, **kwargs) -> pandas.DataFrame:
    total_size = dataset.shape[0]
    if isinstance(size, float):
        sample_size = int(total_size*size)
    else:
        sample_size = size
    if method == 'random':
        return dataset.sample(n=sample_size)
    elif method == 'stratified-random':
        if 'dependent_variable' not in kwargs:
            raise ValueError(
                "Method 'stratified-random' requires parameter 'dependent_variable' specified.")
        D = dataset.copy()
        dependent_variable = kwargs.get('dependent_variable')
        sampling_ratio = float(sample_size)/total_size
        _, bins = np.histogram(D[dependent_variable], bins='auto')
        D['bin'] = pandas.cut(D[dependent_variable], bins)
        summarization_df = D.groupby('bin', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_ratio, replace=False))
        summarization_df.drop(columns=['bin'], inplace=True)
        return summarization_df

    elif method == 'diversity':
        import diversipy
        summarization = diversipy.subset.psa_select(
            dataset.to_numpy(), sample_size)
        summarization_df = pandas.DataFrame(
            data=summarization, columns=dataset.columns)
        return summarization_df
    elif method == 'kdtree':
        summarization = get_kdtree_samples(dataset.to_numpy(), sample_size)
        summarization_df = pandas.DataFrame(
            data=summarization, columns=dataset.columns)
        return summarization_df
    elif method == 'cover-tree':
        summarization = get_covertree_samples(dataset.to_numpy(), sample_size)
        summarization_df = pandas.DataFrame(
            data=summarization, columns=dataset.columns)
        return summarization_df
    elif method == 'tree-guided':
        trained_models = model_definition['trained_model']['ensemble']['trained_models']
        forest = Forest(trained_models)
        sampling_ratio = float(sample_size)/total_size
        summarization_df = batch_sampling(
            forest=forest, D=dataset, sampling_ratio=sampling_ratio, method='tree-guided')
        return summarization_df


class DataStream():
    def __init__(self, dataset: pandas.DataFrame, tree: Tree, sample_ratio: float):
        self.dataset = dataset
        self.current_index = 0
        self.tree = tree
        self.sample_ratio = sample_ratio
        self._actual_samples = {}

    def length(self) -> int:
        return self.dataset.shape[0]-self.current_index

    def process_push(self):
        node = self.tree.nodes[self.tree.get_leaf_id(
            self.dataset.loc[self.current_index])]
        # print("Push {}".format(node.id))
        if node.id in self._actual_samples:
            self._actual_samples[node.id] += 1
        else:
            self._actual_samples[node.id] = 1

    def process_pop(self, sample_id: int):
        node = self.tree.nodes[self.tree.get_leaf_id(
            self.dataset.loc[sample_id])]
        # print("Pop {}".format(node.id))
        self._actual_samples[node.id] -= 1

    def actual_samples(self, node_id: int):
        return self._actual_samples[node_id]/self.sample_ratio

    def desired_samples(self, node_id: int):
        return self.tree.nodes[node_id].number_samples

    def weight(self, H: list) -> float:
        if len(H) == 0:
            return 1.0
        node = self.tree.nodes[self.tree.get_leaf_id(
            self.dataset.loc[self.current_index])]
        if node.id not in self._actual_samples:
            return 1.0

        if self.desired_samples(node.id) > self.actual_samples(node.id):
            weight = (self.desired_samples(node.id) -
                      self.actual_samples(node.id))/self.desired_samples(node.id)
            return weight
        else:
            return 0.0
        # num_samples = node.number_samples
        # return float(num_samples)
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
    """Algorithm A-ExpJ"""
    H = []  # min priority queue
    while not S.empty() and len(H) < k:
        weight = S.weight(H)
        if weight > 0:
            r = random.random()**(1.0/weight)
        else:
            r = -random.random()
        heapq.heappush(H, (r, S.current(), S.current_index))
        S.process_push()
        S.next()
    Hmin = H[0][0]
    X = np.log(random.random())/np.log(Hmin)
    while not S.empty():
        X -= S.weight(H)
        if X <= 0:
            weight = S.weight(H)
            if weight > 0:
                t = Hmin**weight
                r = random.uniform(t, 1)**(1.0/weight)
            else:
                r = -random.random()
            el = heapq.heappop(H)
            S.process_pop(el[2])
            heapq.heappush(H, (r, S.current(), S.current_index))
            S.process_push()
            Hmin = H[0][0]
            X = np.log(random.random())/np.log(Hmin)
        S.next()
    return pandas.DataFrame(data=map(itemgetter(1), H), columns=S.columns())


def get_chi2(tree: Tree, sample: pandas.DataFrame):
    observed_frequencies = {}
    for row in sample.iterrows():
        leaf_id = tree.get_leaf_id(row[1].to_dict())
        if leaf_id in observed_frequencies:
            observed_frequencies[leaf_id] += 1.0
        else:
            observed_frequencies[leaf_id] = 1.0

    expected_frequencies = tree.get_leaves_num_samples()

    observed_stats = []
    expected_stats = []
    for k, v in observed_frequencies.items():
        if v >= 5:  # only look at the leaves with 5 samples and more to get correct test results
            observed_stats.append(float(v))
            expected_stats.append(float(expected_frequencies[k]))
    expected_stats = expected_stats / \
        np.sum(expected_stats)*np.sum(observed_stats)

    res = scipy.stats.power_divergence(
        observed_stats, expected_stats, lambda_="pearson")
    return res.pvalue


def get_mean_chi2(forest: Forest, sample: pandas.DataFrame):
    pvals = []
    # for tree_idx in trees:
    for tree_idx in range(1, len(forest.trees)):
        tree = forest.trees[tree_idx]
        pvals.append(get_chi2(tree, sample))
    return np.mean(pvals), np.array(pvals)


def batch_sampling(forest: Forest, D: pandas.DataFrame, sampling_ratio: float, method: str = 'tree-guided') -> pandas.DataFrame:
    # configuration
    n = D.shape[0]  # dataset size
    k = int(n*sampling_ratio)  # samples size
    num_trees = len(forest.trees)  # number of trees in the forest
    b = int(num_trees*sampling_ratio)  # number of batches
    batch_size = int(n/b)  # batch size
    samples_from_batch = int(k/b)

    sample = pandas.DataFrame(columns=D.columns)
    for batch_index in range(b):
        # Take a batch of data points of size 𝑛/𝑏
        batch = pandas.DataFrame(
            D.iloc[batch_index*batch_size:(batch_index+1)*batch_size])
        batch.reset_index(inplace=True, drop=True)

        if method == 'tree-guided':
            # Select a tree 𝑡 from the forest at random
            t = random.randrange(1, num_trees)
            S = DataStream(batch, forest.trees[t], float(samples_from_batch)/n)
            batch_sample = reservoir_sample_with_jumps(S, samples_from_batch)
        elif method == 'random':
            batch_sample = batch.sample(n=samples_from_batch)
        else:
            raise ValueError(
                'Invalid argument: "method" should be "tree-guided" or "random".')
        sample = sample.append(batch_sample, ignore_index=True)
    return sample
