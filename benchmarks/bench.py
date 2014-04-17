import numpy as np
from time import time

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.utils import check_random_state


def leaves(forest):
    count = 0.0

    for tree in forest:
        count += (tree.tree_.children_left == -1).sum()

    return count / forest.n_estimators


def average_depth(forest):
    avg = 0.0

    for tree in forest:
        depths = np.zeros(tree.tree_.node_count)
        queue = [(0,0)]

        while len(queue) > 0:
            node, depth = queue.pop(0)
            depths[node] = depth

            if tree.tree_.children_left[node] != -1:
                queue.append((tree.tree_.children_left[node], depth + 1))
                queue.append((tree.tree_.children_right[node], depth + 1))

        leaves = tree.tree_.children_left == -1
        avg += np.dot(depths[leaves], tree.tree_.weighted_n_node_samples[leaves]) / tree.tree_.weighted_n_node_samples[0]

    return avg / forest.n_estimators


def bench_artificial(estimator, generator, n_train=200, n_test=10000, n_repeats=10, random_state=None):
    random_state = check_random_state(random_state)

    X_test, y_test = generator(n_samples=n_test, n_features=5, random_state=random_state)

    results = {}
    results["fit_time"] = []
    results["predict_time"] = []
    results["score"] = []
    results["leaves"] = []
    results["average_depth"] = []

    for i in range(n_repeats):
        X_train, y_train = generator(n_samples=n_train, n_features=5, random_state=random_state)
        estimator.set_params(random_state=random_state.randint(0, 1000))

        X_train = np.asarray(X_train, order="f", dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)

        t0 = time()
        estimator.fit(X_train, y_train)
        t1 = time()
        results["fit_time"].append(t1 - t0)

        t0 = time()
        y_hat = estimator.predict(X_test)
        t1 = time()
        results["predict_time"].append((t1 - t0) / n_test)

        results["score"].append(estimator.score(X_test, y_test))
        results["leaves"].append(leaves(estimator))
        results["average_depth"].append(average_depth(estimator))

    return results


if __name__ == "__main__":
    from sklearn.datasets import make_friedman1

    print "RandomForestRegressor"
    results = bench_artificial(RandomForestRegressor(n_estimators=250, max_features="sqrt"), make_friedman1, random_state=0)
    for k,v in results.items():
        print k, np.mean(v), np.std(v)
    print

    print "ExtraTreesRegressor"
    results = bench_artificial(ExtraTreesRegressor(n_estimators=250, max_features="sqrt"), make_friedman1, random_state=0)
    for k,v in results.items():
        print k, np.mean(v), np.std(v)
    print

