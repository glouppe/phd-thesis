import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn.utils import check_random_state


def make(n_samples, n_features=5, noise_features=5, random_state=None):
    X = check_random_state(random_state).normal(size=(n_samples, n_features+noise_features))
    y = np.sum(X[:, :n_features], axis=1)
    return X, y


# from sklearn.datasets import make_friedman1 as make
# # make = partial(make,)

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor #, PERTRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

n_train = 50
n_test = 600
n_estimators = 10       # number of trees per forest
n_sets = 100             # number of learning sets
n_trees = 50            # number of trees per learning sets, for estimating statistics

estimators = [("PERTRegressor", PERTRegressor),
              ("Bagging", partial(RandomForestRegressor, max_features=1.0, bootstrap=True))]
estimators.extend([("RandomForestRegressor-K=%d" % i, partial(RandomForestRegressor, max_features=i)) for i in range(1, 10+1)])
estimators.extend([("ExtraTreesRegressor-K=%d" % i, partial(ExtraTreesRegressor, max_features=i)) for i in range(1, 10+1)])

estimators = []
estimators.extend([("RandomForestRegressor-M=%d" % i, partial(RandomForestRegressor, n_estimators=i, max_features=1)) for i in range(1, 50+1)])
#estimators.extend([("ExtraTreesRegressor-M=%d" % i, partial(ExtraTreesRegressor, n_estimators=i, max_features=1)) for i in range(1, 50+1)])


train = [make(n_samples=n_train, random_state=i) for i in range(n_sets)]
X_test, y_test = make(n_samples=n_test)

for m in range(1, 50+1):
    n_estimators = m
    estimator = partial(RandomForestRegressor, n_estimators=m, max_features=1)
    method = "RandomForestRegressor-M=%d" % m

    # Compute bias/variance on forest predictions
    forests = []

    for k, (X_train, y_train) in enumerate(train):
        #forests.append(estimator(n_estimators=n_estimators, random_state=k).fit(X_train, y_train))
        forests.append(estimator(random_state=k).fit(X_train, y_train))

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
        #trees.extend(estimator(n_estimators=n_trees, random_state=n_sets+k).fit(X_train, y_train).estimators_)
        trees.extend(RandomForestRegressor(n_estimators=n_trees, max_features=1, random_state=n_sets+k).fit(X_train, y_train).estimators_)

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

    print "%s,%f,%f,%f,%f,%f,%f,%f" % (method, bias.mean()+var.mean(), bias.mean(), var.mean(), rho.mean(), rho.std(), (rho*sigma).mean(), ((1 - rho) / n_estimators * sigma).mean())

    # print "%f (error) = %f (b^2) + %f (var)" % (error, bias_forest.mean(), var_forest.mean())
    # print "%f (error) = %f (b^2) + %f (rho*sigma + (1-rho)/M*sigma)" % (bias.mean()+var.mean(), bias.mean(), var.mean())
    # print "var = %f (rho*sigma) + %f (1-rho)/M*sigma ; rho = %f" % ((rho*sigma).mean(), ((1 - rho) / n_estimators * sigma).mean(), rho.mean())
    # print "---"
