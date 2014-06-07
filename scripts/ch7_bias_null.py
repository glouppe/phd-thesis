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

def generate_strobl_null(n_samples=120):
    X = np.array([v for v in product(range(2),
                                     range(4),
                                     range(10),
                                     range(20),
                                     range(2))]).astype(np.int32)
    X, y = X[:, :-1], X[:, -1]

    indices = np.random.randint(0, X.shape[0], n_samples)
    X, y = X[indices], y[indices].astype(np.int32)
    X = np.hstack((np.random.rand(len(X), 1), X))

    return X, y

# Generate all importances
models = [("TRT", partial(RandomizedID3Ensemble, base_estimator=RandomizedID3Classifier(k=1))),
          ("ETs K=1", partial(ExtraTreesClassifier, max_features=1, criterion="entropy")),
          ("ETs K=3", partial(ExtraTreesClassifier, max_features=3, criterion="entropy")),
          ("ETs K=5", partial(ExtraTreesClassifier, max_features=5, criterion="entropy")),
          ("RF K=1", partial(RandomForestClassifier, max_features=1, bootstrap=True, criterion="entropy")),
          ("RF K=3", partial(RandomForestClassifier, max_features=3, bootstrap=True, criterion="entropy")),
          ("RF K=5", partial(RandomForestClassifier, max_features=5, bootstrap=True, criterion="entropy")),]

n_repeat = 5
r = {}

for i in range(n_repeat):
    print "Iteration", i

    X, y = generate_strobl_null(n_samples=120)
    print entropy(y)

    for name, cls in models:
        f = feature_importances(X, y, cls=cls, n_trees=500)

        if i == 0:
            r[name] = np.array(f)
        else:
            r[name] += np.array(f)

        print name, np.sum(f)

for name in r:
    r[name] /= n_repeat

# Convert to pandas and plot
df = pd.DataFrame(r, index=["X%d" % (i+1) for i in range(X.shape[1])])
df = df.reindex_axis([name for name, _ in models], axis=1)

import brewer2mpl
cmap = brewer2mpl.get_map('RdYlGn', 'diverging', len(r))
df.plot(kind="bar", colormap=cmap.mpl_colormap, legend="best", grid=False)
plt.show()
