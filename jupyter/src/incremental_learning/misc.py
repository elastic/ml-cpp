# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import csv
# import dask
# from dask import dataframe
import heapq
import random
from collections import namedtuple
from operator import itemgetter

import numpy as np
import pandas


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

Row = namedtuple('Row', ['index', 'content'])

class DataStream():
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.current_index = 0
        self._actual_samples = {}

    def __enter__(self,):
        self.csvfile = open(self.dataset)
        self.reader = csv.DictReader(self.csvfile, dialect='unix')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.csvfile.close()
        return False

    def __iter__(self):
        return self

    def __next__(self):
        content = next(self.reader)
        return Row(index = self.reader.line_num, content=content)

    def weight(self, H: list) -> float:
        return 1.0

    def current(self):
        return self.dataset.loc[self.current_index]

    def columns(self):
        return self.reader.fieldnames


def reservoir_sample_with_jumps(dataset: str, k: int):
    """Algorithm A-ExpJ"""
    H = []  # min priority queue
    with DataStream(dataset) as S:
        for row in S:
            weight = S.weight(H)
            if weight > 0:
                r = random.random()**(1.0/weight)
            else:
                r = -random.random()
            heapq.heappush(H, (r, row.content, row.index))
            if len(H) >= k:
                break
    Hmin = H[0][0]
    X = np.log(random.random())/np.log(Hmin)
    with DataStream(dataset) as S:
        for row in S:
            X -= S.weight(H)
            if X <= 0:
                weight = S.weight(H)
                if weight > 0:
                    t = Hmin**weight
                    r = random.uniform(t, 1)**(1.0/weight)
                else:
                    r = -random.random()
                el = heapq.heappop(H)
                heapq.heappush(H, (r, row.content, row.index))
                Hmin = H[0][0]
                X = np.log(random.random())/np.log(Hmin)
    return pandas.DataFrame(data=map(itemgetter(1), H), columns=S.columns())



