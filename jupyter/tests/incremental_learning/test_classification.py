# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import unittest

import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from incremental_learning.job import train
from incremental_learning.storage import download_dataset


class TestClassification(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = 100
        self.dataset_name = 'classification-2d'

        download_dataset(self.dataset_name)
        x = np.random.random(1000).reshape((-1, 2))
        y = x[:, 1] < (np.sin(x[:, 0]*np.pi/2*4)+1)/2
        is_training = (x[:, 0] < 0.3) | (x[:, 0] > 0.7) | (
            x[:, 1] < 0.3) | (x[:, 1] > 0.7)

        D = pd.DataFrame(data=x, columns=['x1', 'x2'])
        D['target'] = y
        D['target'].replace({True: 'foo', False: 'bar'}, inplace=True)
        D['training'] = is_training

        self.dataset_train = D.where(D['training'] == True).dropna()
        self.dataset_update = D.where(D['training'] == False).dropna()

    def test_metrics(self) -> None:
        job = train(self.dataset_name, self.dataset_train[[
                    'x1', 'x2', 'target']], verbose=False)
        job.wait_to_complete()

        y_true = self.dataset_train['target']
        classes = np.unique(y_true)
        y_pred = job.get_predictions()
        top_classes = job.get_probabilities()
        # roc_auc_score requires the probability of the "greater" class label
        y_score = np.array([prob[classes[1]] for c, prob in  zip(y_true, top_classes)])
        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        self.assertTrue(accuracy > 0.96, msg = "Low accuracy {}".format(accuracy))
        actual_roc_auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
        self.assertTrue(actual_roc_auc > 0.98, msg="Low roc auc score {}".format(actual_roc_auc))


if __name__ == '__main__':
    unittest.main()
