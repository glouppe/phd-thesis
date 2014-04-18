import numbers
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import six


# R randomForest ==============================================================

from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
rf = importr('randomForest')

class RRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10,
                       min_samples_leaf=1,
                       max_features="auto",
                       random_state=None):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        # Check params
        self.n_features_ = X.shape[1]

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        params = {}
        params["mtry"] = max_features
        params["ntrees"] = self.n_estimators
        params["nodesize"] = self.min_samples_leaf

        # Convert data
        self.classes_ = np.unique(y)
        y = np.searchsorted(self.classes_, y) + 1
        X = numpy2ri(X)
        y = ro.FactorVector(numpy2ri(y))

        # Run
        self.model_ = rf.randomForest(X, y, **params)

        return self

    def predict(self, X):
        X = numpy2ri(X)
        pred = rf.predict_randomForest(self.model_, X)
        # R maps class labels
        pred = np.array(pred, dtype=np.int32) - 1
        return self.classes_[pred]


# OpenCV Random Trees =========================================================

import cv2

class OpenCVRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10,
                       max_depth=None,
                       min_samples_split=2,
                       max_features="auto",
                       random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        # Check params
        self.n_features_ = X.shape[1]

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        params = {}
        params["nactive_vars"] = max_features
        params["max_num_of_trees_in_the_forest"] = self.n_estimators
        params["min_sample_count"] = self.min_samples_split
        params["calc_var_importance"] = False
        params["max_depth"] = 10E9 if self.max_depth is None else self.max_depth
        params["use_surrogates"] = False
        params["termcrit_type"] = (cv2.TERM_CRITERIA_MAX_ITER, self.n_estimators, 1)

        # Convert data
        self.classes_ = np.unique(y)
        y = np.searchsorted(self.classes_, y)
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Run
        self.model_ = cv2.RandomTrees()
        self.model_.train(X, cv2.TERM_CRITERIA_MAX_ITER, y, params=params)

        return self

    def predict(self, X):
        X = X.astype(np.float32)
        pred = np.zeros(len(X))

        for i in len(X):
            pred[i] = self.model_.predict(X[i])

        pred = pred.astype(np.int32)

        return self.classes_[pred]
