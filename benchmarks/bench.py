import numpy as np
import json
from copy import deepcopy
from functools import partial
from itertools import product
from time import time

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.utils import check_random_state

from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3
from sklearn.metrics.scorer import accuracy_scorer, roc_auc_scorer
from sklearn.metrics.scorer import mean_squared_error_scorer, r2_scorer

from data import make_waveforms, make_twonorm, make_threenorm, make_ringnorm


# Utils =======================================================================

def leaves(forest):
    count = 0.0

    for tree in forest:
        count += (tree.tree_.children_left == -1).sum()

    return count / forest.n_estimators


def average_depth(forest):
    avg = 0.0

    for tree in forest:
        tree = tree.tree_
        depths = np.zeros(tree.node_count)
        queue = [(0,0)]

        while len(queue) > 0:
            node, depth = queue.pop(0)
            depths[node] = depth

            if tree.children_left[node] != -1:
                queue.append((tree.children_left[node], depth + 1))
                queue.append((tree.children_right[node], depth + 1))

        leaves = tree.children_left == -1
        avg += np.dot(depths[leaves], tree.weighted_n_node_samples[leaves]) / tree.weighted_n_node_samples[0]

    return avg / forest.n_estimators


def load_npz(filename, random_state=None):
    random_state = check_random_state(random_state)
    data = np.load(filename)

    X = data["X_train"]
    y = data["y_train"].flatten()

    try:
        if len(X_valid) == len(y_valid):
            X_valid = data["X_valid"]
            y_valid = data["y_valid"].flatten()

            X = np.vstack((X, X_valid))
            y = np.hstack((y, y_valid))

    except:
        pass

    try:
        if len(X_test) == len(y_test):
            X_test = data["X_test"]
            y_test = data["y_test"].flatten()

            X = np.vstack((X, X_test))
            y = np.hstack((y, y_test))

    except:
        pass

    indices = random_state.permutation(X.shape[0])
    half = int(0.5 * X.shape[0])
    threequarters = int(0.75 * X.shape[0])

    X_train = X[indices[:threequarters]]
    y_train = y[indices[:threequarters]]
    # X_valid = X[indices[half:threequarters]]
    # y_valid = y[indices[half:threequarters]]
    X_test = X[indices[threequarters:]]
    y_test = y[indices[threequarters:]]

    return X_train, y_train, X_test, y_test


# Benchs ======================================================================

def bench_artificial(estimator, generator, scorers, n_train=500, n_test=10000, n_repeats=10, random_state=None):
    random_state = check_random_state(random_state)

    X_test, y_test = generator(n_samples=n_test, random_state=random_state)

    results = {}
    results["time_fit"] = []
    results["time_predict"] = []
    results["leaves"] = []
    results["average_depth"] = []
    results["n_train"] = []
    results["n_test"] = []
    results["n_features"] = []

    for scorer in scorers:
        results["score_%s" % str(scorer)] = []

    for i in range(n_repeats):
        X_train, y_train = generator(n_samples=n_train, random_state=random_state)
        estimator.set_params(random_state=random_state.randint(0, 1000))

        X_train = np.asarray(X_train, order="f", dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)

        t0 = time()
        estimator.fit(X_train, y_train)
        t1 = time()
        results["time_fit"].append(t1 - t0)

        t0 = time()
        y_hat = estimator.predict(X_test)
        t1 = time()
        results["time_predict"].append((t1 - t0))

        for scorer in scorers:
            try:
                results["score_%s" % str(scorer)].append(scorer(estimator, X_test, y_test))
            except:
                pass

        results["leaves"].append(leaves(estimator))
        results["average_depth"].append(average_depth(estimator))
        results["n_train"].append(len(X_train))
        results["n_test"].append(len(X_test))
        results["n_features"].append(X_train.shape[1])

    return results


def bench_npy(estimator, filename, scorers, n_repeats=10, random_state=None):
    random_state = check_random_state(random_state)

    results = {}
    results["time_fit"] = []
    results["time_predict"] = []
    results["n_train"] = []
    results["n_test"] = []
    results["n_features"] = []

    for scorer in scorers:
        results["score_%s" % str(scorer)] = []

    for i in range(n_repeats):
        X_train, y_train, X_test, y_test = load_npz(filename, random_state=random_state)
        estimator.set_params(random_state=random_state.randint(0, 1000))

        X_train = np.asarray(X_train, order="f", dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)

        try:
            t0 = time()
            estimator.fit(X_train, y_train)
            t1 = time()
            results["time_fit"].append(t1 - t0)

            t0 = time()
            y_hat = estimator.predict(X_test)
            t1 = time()
            results["time_predict"].append((t1 - t0))

        except:
            pass

        for scorer in scorers:
            try:
                results["score_%s" % str(scorer)].append(scorer(estimator, X_test, y_test))
            except:
                pass

        results["n_train"].append(len(X_train))
        results["n_test"].append(len(X_test))
        results["n_features"].append(X_train.shape[1])

        if results["time_fit"][-1] > 3600 and i >= 2:
            break

    return results


# Runs ========================================================================

# Regression ------------------------------------------------------------------

def run_artificial_regression_n_estimators():
    estimators = [("RandomForestRegressor", RandomForestRegressor(n_estimators=250, max_features="sqrt")),
                  ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=250, max_features="sqrt"))]
    generators = [make_friedman1, make_friedman2, make_friedman3]
    scorers = [mean_squared_error_scorer, r2_scorer]

    i = 1
    for n_estimators in map(int, np.logspace(0, 3, num=10)):
        for (estimator_name, estimator), generator in product(estimators, generators):
            print i, n_estimators, estimator_name, generator.__name__

            estimator = deepcopy(estimator)
            estimator.set_params(n_estimators=n_estimators)

            run = {}
            run["estimator"] = estimator_name
            run["generator"] = generator.__name__
            run["params"] = deepcopy(estimator.get_params(deep=False))
            run["stats"] = bench_artificial(estimator, generator, scorers=scorers, random_state=0)

            with open("output/n_estimators_%s_%s_%d_%d.json" % (estimator_name, generator.__name__, n_estimators, i), "w") as fd:
                json.dump(run, fd)

            i += 1


def run_artificial_regression_max_features():
    estimators = [("RandomForestRegressor", RandomForestRegressor(n_estimators=250, max_features="sqrt")),
                  ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=250, max_features="sqrt"))]
    generators = [make_friedman1]
    scorers = [mean_squared_error_scorer, r2_scorer]

    i = 1
    for max_features in range(1, 10+1):
        for (estimator_name, estimator), generator in product(estimators, generators):
            print i, max_features, estimator_name, generator.__name__

            estimator = deepcopy(estimator)
            estimator.set_params(max_features=max_features)

            run = {}
            run["estimator"] = estimator_name
            run["generator"] = generator.__name__
            run["params"] = deepcopy(estimator.get_params(deep=False))
            run["stats"] = bench_artificial(estimator, generator, scorers=scorers, random_state=0)

            with open("output/max_features_%s_%s_%d_%d.json" % (estimator_name, generator.__name__, max_features, i), "w") as fd:
                json.dump(run, fd)

            i += 1


def run_artificial_regression_bootstrap():
    estimators = [("RandomForestRegressor", RandomForestRegressor(n_estimators=250, max_features="sqrt")),
                  ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=250, max_features="sqrt"))]
    generators = [make_friedman1, make_friedman2, make_friedman3]
    scorers = [mean_squared_error_scorer, r2_scorer]

    i = 1
    for bootstrap in [True, False]:
        for (estimator_name, estimator), generator in product(estimators, generators):
            print i, bootstrap, estimator_name, generator.__name__

            estimator = deepcopy(estimator)
            estimator.set_params(bootstrap=bootstrap)

            run = {}
            run["estimator"] = estimator_name
            run["generator"] = generator.__name__
            run["params"] = deepcopy(estimator.get_params(deep=False))
            run["stats"] = bench_artificial(estimator, generator, scorers=scorers, random_state=0)

            with open("output/bootstrap_%s_%s_%s_%d.json" % (estimator_name, generator.__name__, bootstrap, i), "w") as fd:
                json.dump(run, fd)

            i += 1


def run_artificial_regression_n_train():
    estimators = [("RandomForestRegressor", RandomForestRegressor(n_estimators=250, max_features="sqrt")),
                  ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=250, max_features="sqrt"))]
    generators = [make_friedman1, make_friedman2, make_friedman3]
    scorers = [mean_squared_error_scorer, r2_scorer]

    i = 1
    for n_train in map(int, np.logspace(0, 4, num=10)):
        for (estimator_name, estimator), generator in product(estimators, generators):
            print i, n_train, estimator_name, generator.__name__

            estimator = deepcopy(estimator)

            run = {}
            run["estimator"] = estimator_name
            run["generator"] = generator.__name__
            run["params"] = deepcopy(estimator.get_params(deep=False))
            run["stats"] = bench_artificial(estimator, generator, n_train=n_train, scorers=scorers, random_state=0)

            with open("output/n_train_%s_%s_%d_%d.json" % (estimator_name, generator.__name__, n_train, i), "w") as fd:
                json.dump(run, fd)

            i += 1


def run_artificial_regression_n_features():
    estimators = [("RandomForestRegressor", RandomForestRegressor(n_estimators=250, max_features="sqrt")),
                  ("ExtraTreesRegressor", ExtraTreesRegressor(n_estimators=250, max_features="sqrt"))]
    generators = [make_friedman1]
    scorers = [mean_squared_error_scorer, r2_scorer]

    i = 1
    for n_features in map(int, np.logspace(0, 3, num=10)):
        n_features += 5

        for (estimator_name, estimator), generator in product(estimators, generators):
            print i, n_features, estimator_name, generator.__name__

            estimator = deepcopy(estimator)

            run = {}
            run["estimator"] = estimator_name
            run["generator"] = generator.__name__
            run["params"] = deepcopy(estimator.get_params(deep=False))
            run["stats"] = bench_artificial(estimator, partial(generator, n_features=n_features), scorers=scorers, random_state=0)

            with open("output/n_features_%s_%s_%d_%d.json" % (estimator_name, generator.__name__, n_features, i), "w") as fd:
                json.dump(run, fd)

            i += 1


# Classification --------------------------------------------------------------

def run_artificial_classification_n_estimators():
    estimators = [("RandomForestClassifier", RandomForestClassifier(n_estimators=250, max_features="sqrt")),
                  ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=250, max_features="sqrt"))]
    generators = [make_hastie_10_2, make_waveforms, make_twonorm, make_threenorm, make_ringnorm]
    scorers = [accuracy_scorer, roc_auc_scorer]

    i = 1
    for n_estimators in map(int, np.logspace(0, 3, num=10)):
        for (estimator_name, estimator), generator in product(estimators, generators):
            print i, n_estimators, estimator_name, generator.__name__

            estimator = deepcopy(estimator)
            estimator.set_params(n_estimators=n_estimators)

            run = {}
            run["estimator"] = estimator_name
            run["generator"] = generator.__name__
            run["params"] = deepcopy(estimator.get_params(deep=False))
            run["stats"] = bench_artificial(estimator, generator, scorers=scorers, random_state=0)

            with open("output/n_estimators_%s_%s_%d_%d.json" % (estimator_name, generator.__name__, n_estimators, i), "w") as fd:
                json.dump(run, fd)

            i += 1


def run_artificial_classification_max_features():
    estimators = [("RandomForestClassifier", RandomForestClassifier(n_estimators=250, max_features="sqrt")),
                  ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=250, max_features="sqrt"))]
    generators = [make_waveforms, make_twonorm, make_threenorm, make_ringnorm]
    scorers = [accuracy_scorer, roc_auc_scorer]

    i = 1
    for max_features in range(1, 20+1):
        for (estimator_name, estimator), generator in product(estimators, generators):
            print i, max_features, estimator_name, generator.__name__

            estimator = deepcopy(estimator)
            estimator.set_params(max_features=max_features)

            run = {}
            run["estimator"] = estimator_name
            run["generator"] = generator.__name__
            run["params"] = deepcopy(estimator.get_params(deep=False))
            run["stats"] = bench_artificial(estimator, generator, scorers=scorers, random_state=0)

            with open("output/max_features_%s_%s_%d_%d.json" % (estimator_name, generator.__name__, max_features, i), "w") as fd:
                json.dump(run, fd)

            i += 1


def run_artificial_classification_bootstrap():
    estimators = [("RandomForestClassifier", RandomForestClassifier(n_estimators=250, max_features="sqrt")),
                  ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=250, max_features="sqrt"))]
    generators = [make_hastie_10_2, make_waveforms, make_twonorm, make_threenorm, make_ringnorm]
    scorers = [accuracy_scorer, roc_auc_scorer]

    i = 1
    for bootstrap in [True, False]:
        for (estimator_name, estimator), generator in product(estimators, generators):
            print i, bootstrap, estimator_name, generator.__name__

            estimator = deepcopy(estimator)
            estimator.set_params(bootstrap=bootstrap)

            run = {}
            run["estimator"] = estimator_name
            run["generator"] = generator.__name__
            run["params"] = deepcopy(estimator.get_params(deep=False))
            run["stats"] = bench_artificial(estimator, generator, scorers=scorers, random_state=0)

            with open("output/bootstrap_%s_%s_%s_%d.json" % (estimator_name, generator.__name__, bootstrap, i), "w") as fd:
                json.dump(run, fd)

            i += 1


def run_artificial_classification_n_train():
    estimators = [("RandomForestClassifier", RandomForestClassifier(n_estimators=250, max_features="sqrt")),
                  ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=250, max_features="sqrt"))]
    generators = [make_hastie_10_2, make_waveforms, make_twonorm, make_threenorm, make_ringnorm]
    scorers = [accuracy_scorer, roc_auc_scorer]

    i = 1
    for n_train in map(int, np.logspace(0, 4, num=10)):
        for (estimator_name, estimator), generator in product(estimators, generators):
            print i, n_train, estimator_name, generator.__name__

            estimator = deepcopy(estimator)

            run = {}
            run["estimator"] = estimator_name
            run["generator"] = generator.__name__
            run["params"] = deepcopy(estimator.get_params(deep=False))
            run["stats"] = bench_artificial(estimator, generator, n_train=n_train, scorers=scorers, random_state=0)

            with open("output/n_train_%s_%s_%d_%d.json" % (estimator_name, generator.__name__, n_train, i), "w") as fd:
                json.dump(run, fd)

            i += 1


# Datatsets -------------------------------------------------------------------

from wrapper import RRandomForestClassifier
from wrapper import OpenCVRandomForestClassifier

def run_npy_default(prefix="/home/gilles/PhD/db/data/"):
    datasets = [#"diabetes.npz", "dig44.npz", "ionosphere.npz", "pendigits.npz",
                #"letter.npz", "liver.npz", "musk2.npz", "ring-norm.npz", "satellite.npz",
                #"segment.npz", "sonar.npz", "spambase.npz", "two-norm.npz", "vehicle.npz",
                #"vowel.npz", "waveform.npz",
                "cifar10.npz", "mnist3vs8.npz", "mnist4vs9.npz", "mnist.npz",
                "isolet.npz", "arcene.npz", "breast2.npz", "madelon.npz", "marti0.npz",
                "reged0.npz", "secom.npz", "tis.npz", "sido0.npz"]

    estimators = [("RandomForestClassifier", RandomForestClassifier(n_estimators=250, max_features="sqrt")),
                  ("ExtraTreesClassifier", ExtraTreesClassifier(n_estimators=250, max_features="sqrt")),
                  ("R-randomForest", RRandomForestClassifier(n_estimators=250, max_features="sqrt")),
                  ("OpenCV", OpenCVRandomForestClassifier(n_estimators=250, max_features="sqrt"))]
    scorers = [accuracy_scorer, roc_auc_scorer]

    i = 1
    for dataset, (estimator_name, estimator) in product(datasets, estimators):
        print i, estimator_name, dataset

        estimator = deepcopy(estimator)

        run = {}
        run["estimator"] = estimator_name
        run["generator"] = dataset
        run["params"] = deepcopy(estimator.get_params(deep=False))
        run["stats"] = bench_npy(estimator, "%s%s" % (prefix, dataset), scorers=scorers, random_state=0)

        with open("output/default_%s_%s_%d.json" % (estimator_name, dataset, i), "w") as fd:
            json.dump(run, fd)

        i += 1


if __name__ == "__main__":
    # Test on artifical data ==================================================
    # REGRESSION

    # run_artificial_regression_n_estimators()
    # run_artificial_regression_max_features()
    # run_artificial_regression_bootstrap()
    # run_artificial_regression_n_train()
    # run_artificial_regression_n_features()

    # CLASSIFICATION

    # run_artificial_classification_n_estimators()
    # run_artificial_classification_max_features()
    # run_artificial_classification_bootstrap()
    # run_artificial_classification_n_train()


    # Test on real data =======================================================

    # CLASSIFICATION

    run_npy_default(prefix="/home/glouppe/db/")
