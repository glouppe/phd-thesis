import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
cmap = brewer2mpl.get_map('RdYlGn', 'diverging', 7).mpl_colors
#cmap = [(0, 0, 1.0), (1.0, 0, 0)]

def feature_importances(X, y, n_trees=500):
    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=n_trees, max_features=1, criterion="entropy").fit(X, y)
    imp = np.zeros(X.shape[1])

    for tree in clf.estimators_:
        imp += tree.tree_.compute_feature_importances(normalize=False)
    imp = imp / n_trees
    return imp

def plot_with_dupplicate(X, y, duplicate=0, n_copies=10):
    n_features = X.shape[1]
    all_importances = []
    X_new = np.hstack([X] + [X[:, duplicate:duplicate+1] for i in range(n_copies)])

    for i in range(n_copies+1):
        all_importances.append(feature_importances(X_new[:, :n_features + i], y)[:n_features])

    all_importances = np.array(all_importances)

    for m in range(n_features):
        plt.plot(range(n_copies+1), all_importances[:, m], "o-", label="X%d" % (m+1), color=cmap[m])

    plt.title("Adding copies of X%d" % (duplicate+1))
    plt.legend(loc="best")
    plt.show()

from demo import make_led
X, y = make_led()
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([1, 0, 0, 1])
plot_with_dupplicate(X, y, duplicate=4, n_copies=100)
