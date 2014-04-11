import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.random.rand(300, 2)
y = (X[:,0] > 0.3) & (X[:,0] < 0.7) & (X[:,1] > 0.3) & (X[:,1] < 0.7)

# randomly flips some labels
mask = np.random.permutation(len(X))[:5]
y[mask] = ~y[mask]

X_c1 = X[y == 0]
plt.scatter(X_c1[:, 0], X_c1[:, 1], color=(1.0, 0, 0))

X_c2 = X[y == 1]
plt.scatter(X_c2[:, 0], X_c2[:, 1], color=(0, 0, 1.0))

# decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_leaf_nodes=5).fit(X, y)
print "children_left =", clf.tree_.children_left
print "children_right =", clf.tree_.children_right
print "feature =", clf.tree_.feature
print "threshold =", clf.tree_.threshold
print "impurity =", clf.tree_.impurity
print "n_samples =", clf.tree_.n_node_samples
print "value =", clf.tree_.value

plt.show()
