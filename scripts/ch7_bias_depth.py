import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import brewer2mpl

from itertools import product
from functools import partial
from demo import entropy

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from ID3 import RandomizedID3Classifier, RandomizedID3Ensemble

import brewer2mpl
cmap = brewer2mpl.get_map('RdYlGn', 'diverging', 5).mpl_colors


def feature_importances(X, y, cls, n_trees=500):
    clf = cls(n_estimators=n_trees).fit(X, y)

    if isinstance(clf, RandomizedID3Ensemble):
        imp = np.sum(clf.feature_importances_, axis=1)

    else:
        imp = np.zeros(X.shape[1])

        for tree in clf.estimators_:
            imp += tree.tree_.compute_feature_importances(normalize=False)

        imp = imp / n_trees

    return imp

def generate_strobl_power(n_samples=120, relevance=0.2):
    X = np.array([v for v in product(range(2), range(4), range(10), range(20))]).astype(np.int32)
    X = np.hstack((np.random.rand(len(X), 1), X))

    y = np.zeros(len(X))
    mask = (X[:, 1] == 1)
    y[mask] = np.random.rand(mask.sum()) < 0.5-relevance
    y[~mask] = np.random.rand((~mask).sum()) < 0.5+relevance

    indices = np.random.permutation(X.shape[0])[:n_samples]
    return X[indices], y[indices].astype(np.int32)

    return X, y

# Generate all importances
#cls = partial(ExtraTreesClassifier, max_features=1, criterion="entropy")
cls = partial(RandomForestClassifier, max_features=5, criterion="entropy")

relevances = [0.0, 0.1, 0.2, 0.3]
depths = range(1, 10)


for i, relevance in enumerate(relevances):
    imp_all = []

    for n in range(10):
        imp = []
        X, y = generate_strobl_power(relevance=relevance)

        for q in depths:
            c = partial(cls, max_depth=q)
            imp.append(feature_importances(X, y, cls=c))

        imp = np.array(imp)
        imp_all.append(imp)

    imp = np.mean(imp_all, axis=0)

    for q in range(imp.shape[0]):
        imp[q] /= np.sum(imp[q, :])

    plt.subplot(2, 2, i + 1)

    for j in range(X.shape[1]):
        plt.plot(depths, imp[:, j], "o-", label="$X_%d$" % j, color=cmap[j])

    plt.ylim([0., 1.0])
    plt.title("Relevance = %.1f" % relevance)

    if i == 0:
        plt.legend(loc="best")

plt.show()
