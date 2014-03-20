import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from sklearn.datasets import make_friedman1 as make
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

TotallyRandomizedTreesRegressor = partial(ExtraTreesRegressor, max_features=1)
BaggingRegressor = partial(RandomForestRegressor, max_features=1.0, bootstrap=True)

estimators = [partial(RandomForestRegressor, max_features=i) for i in range(1, 10+1)]

n_train = 20
n_test = 500
n_estimators = 10       # number of trees per forest

n_sets = 200            # number of learning sets
n_trees = 200           # number of trees per learning sets, for estimating statistics

train = [make(n_samples=n_train, n_features=10, random_state=i) for i in range(n_sets)]
X_test, y_test = make(n_samples=n_test, n_features=10)

for estimator in estimators:
    # Compute bias/variance on forest predictions
    forests = []

    for k, (X_train, y_train) in enumerate(train):
        forests.append(estimator(n_estimators=n_estimators, random_state=k).fit(X_train, y_train))

    pred_forest = np.zeros((n_test, n_sets))

    error = 0.0
    for k, forest in enumerate(forests):
        pred_forest[:, k] = forest.predict(X_test)
        error += mean_squared_error(y_test, pred_forest[:, k])
    error /= n_sets

    bias_forest = (y_test - np.mean(pred_forest, axis=1)) ** 2
    var_forest = np.var(pred_forest, axis=1)

    # Estimate bias/variance from tree predictions
    trees = []

    for k, (X_train, y_train) in enumerate(train):
        trees.extend(estimator(n_estimators=n_trees, random_state=n_sets+k).fit(X_train, y_train).estimators_)

    pred_trees = np.zeros((n_test, n_sets * n_trees))

    for m, tree in enumerate(trees):
        pred_trees[:, m] = tree.predict(X_test)

    mu = np.mean(pred_trees, axis=1)
    sigma = np.var(pred_trees, axis=1)
    rho = np.zeros(n_test)

    for i in range(n_test):
        e_prod = 0.0
        for k in range(n_sets):
            p = pred_trees[i, k*n_trees:(k+1)*n_trees]
            p = p.reshape((n_trees, 1))
            e_prod += np.dot(p, p.T).mean()
        e_prod /= n_sets
        rho[i] = (e_prod - mu[i]**2) / sigma[i]

    bias = (y_test - mu) ** 2
    var = rho * sigma + (1 - rho) / n_estimators * sigma

    print "%f (error) = %f (b^2) + %f (var)" % (error, bias_forest.mean(), var_forest.mean())
    print "%f (error) = %f (b^2) + %f (rho*sigma + (1-rho)/M*sigma)" % (bias.mean()+var.mean(), bias.mean(), var.mean())
    print "var = %f (rho*sigma) + %f (1-rho)/M*sigma ; rho = %f" % ((rho*sigma).mean(), ((1 - rho) / n_estimators * sigma).mean(), rho.mean())
    print "---"
