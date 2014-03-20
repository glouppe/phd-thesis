import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from scipy.spatial.distance import pdist, squareform

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import MDS


def rf_proximities(forest, X):
    prox = pdist(forest.apply(X), lambda u, v: (u == v).sum()) / forest.n_estimators
    prox = squareform(prox)
    return prox


data = load_digits()
X, y = data.data, data.target

indices = np.argsort(y)
X = X[indices]
y = y[indices]

# X = X[y < 2]
# y = y[y < 2]

forest = RandomForestClassifier(n_estimators=50, n_jobs=2, random_state=1).fit(X, y)
prox = rf_proximities(forest, X)

plt.matshow(prox, cmap="Reds")
plt.show()

model = MDS(dissimilarity="precomputed", n_jobs=2)
coords = model.fit_transform(1. - prox)

n_classes = forest.n_classes_
cm = plt.get_cmap("hsv")
colors =  (cm(1. * i / n_classes) for i in range(n_classes))

for k, c in zip(range(n_classes), colors):
    plt.plot(coords[y == k, 0], coords[y == k, 1], '.', label=k, color=c)

plt.legend(loc="best")
plt.show()
