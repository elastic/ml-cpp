# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

from operator import itemgetter
from typing import Union

import numpy as np
import pandas
import scipy.spatial

from .trees import Forest, Tree

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

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


def get_kdtree_samples(X: np.array, nsamples: int) -> np.array:
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
