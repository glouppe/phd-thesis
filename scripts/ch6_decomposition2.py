import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

from demo import make_led
from ID3 import RandomizedID3Classifier, RandomizedID3Ensemble


def feature_importance_tree(clf):
    def _visit(tree, conditioning):
        conditioning = conditioning + [tree[0]]

        if len(tree) == 2:
            pass

        else:
            for X in conditioning:
                imp[tree[0], X] += tree[1]

            for c in tree[2]:
                _visit(c, conditioning)

    imp = np.zeros((clf.n_features_, clf.n_features_))
    _visit(clf.tree_, [])

    return imp

def feature_importances_ensemble(clf):
    importances = np.zeros((clf.p, clf.p))

    for i, tree in enumerate(clf.estimators_):
        importances += feature_importance_tree(tree)

    importances /= clf.n_estimators

    return importances


n_trees = 1000

X, y = make_led()
n_features = X.shape[1]

clf = RandomizedID3Ensemble(n_estimators=n_trees,
                            base_estimator=RandomizedID3Classifier(k=1)).fit(X, y)

imp = feature_importances_ensemble(clf)
plt.imshow(imp, interpolation="nearest", cmap=cm.gist_heat_r)
plt.show()

