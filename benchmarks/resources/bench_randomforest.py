"""
Benchmark script to bench scikit-learn's RandomForestClassifier
vs. R's randomForest.

It uses rpy2 to call R from python. Timings for randomForest are
pessimistic due to a constant overhead by wrapping numpy matrices
in R data_frames. The effect of the overhead can be reduced
by increasing the number of trees.

Note: make sure the LD_LIBRARY_PATH is set for rpy2::

    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64/R/lib
"""

import numpy as np

from time import time
from functools import wraps
from collections import defaultdict

from sklearn import datasets as sk_datasets
from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro

import pylab as pl

rf = importr('randomForest')

data_path = '/home/pprett/corpora'


class RRandomForestClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, **kargs):
        self.params = kargs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        y = np.searchsorted(self.classes_, y) + 1
        X = numpy2ri(X)
        y = ro.FactorVector(numpy2ri(y))
        self.model_ = rf.randomForest(X, y, **self.params)
        return self

    def predict(self, X):
        X = numpy2ri(X)
        pred = rf.predict_randomForest(self.model_, X)
        # R maps class labels
        pred = np.array(pred, dtype=np.int32) - 1
        return self.classes_[pred]


def repeat(n_repetitions=3):
    def wrap(f):
        def wrapper(*args, **kargs):
            scores = []
            for i in range(n_repetitions):
                scores.append(f(*args, random_state=i, **kargs))
            scores = np.array(scores)
            return scores.mean(axis=0), scores.std(axis=0)
        return wraps(f)(wrapper)
    return wrap


@repeat()
def bench_hastie_10_2(clf, random_state=None):
    X, y = sk_datasets.make_hastie_10_2(random_state=random_state)
    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]
    X_train = np.asarray(X_train, order='f', dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    error_rate = np.mean(pred != y_test)
    return error_rate, train_time, test_time


@repeat()
def bench_random_gaussian(clf, random_state=None):
    rs = check_random_state(random_state)
    shape = (12000, 10)
    X = rs.normal(size=shape).reshape(shape)
    y = ((X ** 2.0).sum(axis=1) > 9.34).astype(np.int32)

    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]
    X_train = np.asarray(X_train, order='f', dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    error_rate = np.mean(pred != y_test)
    return error_rate, train_time, test_time


@repeat()
def bench_spam(clf, random_state=None):
    X = np.loadtxt(data_path + "/spam/spambase.data", delimiter=",")
    y = X[:, -1].ravel()
    X = X[:, :-1]
    f = open(data_path + "/spam/spambase.names")
    feature_names = np.array([l.split(":")[0] for l in f])

    X, y = shuffle(X, y, random_state=random_state)
    X_test, y_test = X[:1536], y[:1536]
    X_train, y_train = X[1536:], y[1536:]
    X_train = np.asarray(X_train, order='f', dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    error_rate = (1.0 - clf.score(X_test, y_test))
    test_time = time() - t0
    return error_rate, train_time, test_time


@repeat()
def bench_madelon(clf, random_state=None):
    X_train = np.loadtxt(data_path + "/madelon/madelon_train.data")
    y_train = np.loadtxt(data_path + "/madelon/madelon_train.labels")
    X_test = np.loadtxt(data_path + "/madelon/madelon_valid.data")
    y_test = np.loadtxt(data_path + "/madelon/madelon_valid.labels")
    X_train = np.asarray(X_train, order='f', dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    error_rate = (1.0 - clf.score(X_test, y_test))
    test_time = time() - t0
    return error_rate, train_time, test_time


@repeat()
def bench_arcene(clf, random_state=None):
    X_train = np.loadtxt(data_path + "/arcene/arcene_train.data")
    y_train = np.loadtxt(data_path + "/arcene/arcene_train.labels")
    X_test = np.loadtxt(data_path + "/arcene/arcene_valid.data")
    y_test = np.loadtxt(data_path + "/arcene/arcene_valid.labels")
    X_train = np.asarray(X_train, order='f', dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    error_rate = (1.0 - clf.score(X_test, y_test))
    test_time = time() - t0
    return error_rate, train_time, test_time


@repeat()
def bench_landsat(clf, random_state=None):
    landsat = sk_datasets.load_landsat()
    X = np.asarray(landsat.data, order='f', dtype=np.float32)
    y = landsat.target
    t0 = time()
    clf.fit(X, y)
    train_time = time() - t0
    t0 = time()
    error_rate = (1.0 - clf.score(X, y))
    test_time = time() - t0
    return error_rate, train_time, test_time


@repeat(1)
def bench_mnist(clf, random_state=None):
    rs = check_random_state(random_state)
    mnist = sk_datasets.fetch_mldata('MNIST original')
    inds = np.arange(len(mnist.data))
    rs.shuffle(inds)
    cut_off = int(0.9 * len(inds))
    train_i = inds[:cut_off]
    test_i = inds[cut_off:]

    X_train = mnist.data[train_i].astype(np.float32)
    y_train = mnist.target[train_i].astype(np.float64)

    X_test = mnist.data[test_i].astype(np.float32)
    y_test = mnist.target[test_i].astype(np.float64)

    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    t0 = time()
    error_rate = (1.0 - clf.score(X_test, y_test))
    test_time = time() - t0
    return error_rate, train_time, test_time


if __name__ == '__main__':
    res = defaultdict(dict)

    clfs = {'r': RRandomForestClassifier(ntree=100, mtry=3, nodesize=1),
            'py': RandomForestClassifier(n_estimators=100, max_features=3,
                                         min_samples_leaf=1,
                                         n_jobs=1)}
    datasets = {'random_gaussian': bench_random_gaussian,
                'spam': bench_spam,
                'madelon': bench_madelon,
                'arcene': bench_arcene,
                'landsat': bench_landsat,
                'hastie_10_2': bench_hastie_10_2}

    for impl, clf in clfs.iteritems():
        for dataset, ds_bench in datasets.iteritems():
            mean, std = ds_bench(clf)
            res[dataset][impl] = (mean, std)

    clfs = {'r': RRandomForestClassifier(ntree=10, mtry=3, nodesize=1),
            'py': RandomForestClassifier(n_estimators=10, max_features=3,
                                         min_samples_leaf=1,
                                         n_jobs=1)}
    datasets = {'mnist': bench_mnist}
    for impl, clf in clfs.iteritems():
        for dataset, ds_bench in datasets.iteritems():
            mean, std = ds_bench(clf)
            res[dataset][impl] = (mean, std)

    for ds in res:
        print('_' * 80)
        print(ds)
        print
        print("%s\t%s\t%s" % (' '*4, 'r'.center(13), 'py'.center(13)))
        for i, metric in enumerate(['score', 'train', 'test']):
            print("%s\t%.4f (%.2f)\t%.4f (%.2f)" %
                  (metric, res[ds]['r'][0][i], res[ds]['r'][1][i],
                   res[ds]['py'][0][i], res[ds]['py'][1][i]))
        print

