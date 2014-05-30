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

def feature_importances(X, y, cls, n_trees=5000):
    clf = cls(n_estimators=n_trees).fit(X, y)

    if isinstance(clf, RandomizedID3Ensemble):
        imp = np.sum(clf.feature_importances_, axis=1)

    else:
        imp = np.zeros(X.shape[1])

        for tree in clf.estimators_:
            imp += tree.tree_.compute_feature_importances(normalize=False)

        imp = imp / n_trees

    return imp

def generate_copy(n1=20, n2=2):
    X = np.array([np.arange(n1), np.arange(n1)]).T
    X[:, 1] = X[:, 0] >= n1/2
    y = X[:, 1]
    return X, y

import brewer2mpl
cmap = [(1., 0, 0), (0, 0, 1)]

r = {}
g = generate_copy

for name, cls in [("ETs", partial(ExtraTreesClassifier, max_features=1, criterion="entropy")),
                  ("RF", partial(RandomForestClassifier, max_features=1, bootstrap=False, criterion="entropy"))]:
    f = []
    for n1 in range(2, 20+1, 2):
        X, y = g(n1=n1, n2=2)
        f.append(feature_importances(X, y, cls=cls))
    r[name] = np.array(f)


models = ["ETs", "RF"]

plt.subplot(1, 2, 1)

for i, name in enumerate(models):
    f = r[name]
    plt.plot(range(2, 20+1, 2), f[:, 0], "o-", label="%s" % name, color=cmap[i])
    plt.ylim([0., 1.0])
    plt.title("$X_1$")
    plt.legend(loc="best")

plt.subplot(1, 2, 2)

for i, name in enumerate(models):
    f = r[name]
    plt.plot(range(2, 20+1, 2), f[:, 1], "o-", label="%s" % name, color=cmap[i])
    plt.title("$X_2$")
    plt.ylim([0., 1.0])

plt.show()

