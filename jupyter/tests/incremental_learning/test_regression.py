from incremental_learning.misc import train, update, evaluate

from sklearn.datasets import make_regression
import pandas as pd
import numpy as np


def test_train_update():
    X, y = make_regression(n_samples = 100, n_features=2, n_informative=2)
    dataset = pd.DataFrame(data=np.concatenate([X,y.reshape(-1,1)], axis=1), columns=['x1', 'x2', 'y'])
    dataset_name = 'test_regression'
    job1 = train(dataset_name, dataset, verbose=False)
    success = job1.wait_to_complete()
    assert success
    assert job1.get_predictions().shape[0] == 100

    X, y = make_regression(n_samples = 50, n_features=2, n_informative=2)
    dataset = pd.DataFrame(data=np.concatenate([X,y.reshape(-1,1)], axis=1), columns=['x1', 'x2', 'y'])
    job2 = update(dataset_name, dataset, job1, verbose=False)
    success = job2.wait_to_complete()
    assert success
    assert job2.get_predictions().shape[0] == 50

def test_train_evaluate():
    X, y = make_regression(n_samples = 100, n_features=2, n_informative=2)
    dataset = pd.DataFrame(data=np.concatenate([X,y.reshape(-1,1)], axis=1), columns=['x1', 'x2', 'y'])
    dataset_name = 'test_regression'
    job1 = train(dataset_name, dataset, verbose=False)
    success = job1.wait_to_complete()
    assert success

    X, y = make_regression(n_samples = 50, n_features=2, n_informative=2)
    dataset = pd.DataFrame(data=np.concatenate([X,y.reshape(-1,1)], axis=1), columns=['x1', 'x2', 'y'])
    job2 = evaluate(dataset_name, dataset, job1, verbose=False)
    success = job2.wait_to_complete()
    assert success
    assert job2.get_predictions().shape[0] == 50

if __name__ == '__main__':
    test_train_evaluate()
