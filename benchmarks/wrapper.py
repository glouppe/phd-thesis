import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import six

from rpy2.robjects.numpy2ri import numpy2ri
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
rf = importr('randomForest')


class RRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10,
                       min_samples_leaf=1,
                       max_features="auto"):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X, y):
        # Check params
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(self.n_features_)))
                else:
                    max_features = self.n_features_
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

