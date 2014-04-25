import numbers
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import six
from sklearn.utils import check_random_state


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
        params["max_depth"] = 100000 if self.max_depth is None else self.max_depth
        params["use_surrogates"] = False
        params["termcrit_type"] = cv2.TERM_CRITERIA_MAX_ITER
        params["term_crit"] = (cv2.TERM_CRITERIA_MAX_ITER, self.n_estimators, 1)
        params["regression_accuracy"] = 0

        var_types = np.array([cv2.CV_VAR_NUMERICAL] * self.n_features_ + [cv2.CV_VAR_CATEGORICAL], np.uint8)

        # Convert data
        self.classes_ = np.unique(y)
        y = np.searchsorted(self.classes_, y)
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Run
        self.model_ = cv2.RTrees()
        self.model_.train(X, cv2.TERM_CRITERIA_MAX_ITER, y, varType=var_types, params=params)

        return self

    def predict(self, X):
        X = X.astype(np.float32)
        pred = np.zeros(len(X))

        for i in range(len(X)):
            pred[i] = self.model_.predict(X[i])

        pred = pred.astype(np.int32)

        return self.classes_[pred]


class OpenCVExtraTreesClassifier(BaseEstimator, ClassifierMixin):
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
        params["max_depth"] = 100000 if self.max_depth is None else self.max_depth
        params["use_surrogates"] = False
        params["termcrit_type"] = cv2.TERM_CRITERIA_MAX_ITER
        params["term_crit"] = (cv2.TERM_CRITERIA_MAX_ITER, self.n_estimators, 1)
        params["regression_accuracy"] = 0

        var_types = np.array([cv2.CV_VAR_NUMERICAL] * self.n_features_ + [cv2.CV_VAR_CATEGORICAL], np.uint8)

        # Convert data
        self.classes_ = np.unique(y)
        y = np.searchsorted(self.classes_, y)
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        # Run
        self.model_ = cv2.ERTrees()
        self.model_.train(X, cv2.TERM_CRITERIA_MAX_ITER, y, varType=var_types, params=params)

        return self

    def predict(self, X):
        X = X.astype(np.float32)
        pred = np.zeros(len(X))

        for i in range(len(X)):
            pred[i] = self.model_.predict(X[i])

        pred = pred.astype(np.int32)

        return self.classes_[pred]


# Weka ========================================================================

import tempfile
import os
from time import time

from weka.classifiers import Classifier

def to_arff(X, y, n_classes, f):
    n_features = X.shape[1]

    f.write("@relation tmp\n")

    for i in range(n_features):
        f.write("@attribute feature%d numeric\n" % i)
    f.write("@attribute class {%s}\n" % ",".join("class%d" % c for c in range(n_classes)))
    f.write("\n")
    f.write("@data\n")

    for i in range(len(X)):
        for v in X[i]:
            f.write(str(v))
            f.write(",")

        if y is None:
            f.write("?\n")
        else:
            f.write("class%d\n" % y[i])

class WekaRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10,
                       max_depth=None,
                       max_features="auto",
                       random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        # Check params
        self.n_features_ = X.shape[1]
        random_state = check_random_state(self.random_state)

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
        params["-I"] = self.n_estimators
        params["-K"] = max_features
        params["-depth"] = 0 if self.max_depth is None else self.max_depth
        params["-no-cv"] = None
        params["-s"] = random_state.randint(1000000)

        # Convert data
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y = np.searchsorted(self.classes_, y)

        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".arff", dir="/dev/shm", delete=False)
        to_arff(X, y, self.n_classes_, tf)
        tf.close()

        # Run
        self.model_ = Classifier(name="weka.classifiers.trees.RandomForest", ckargs=params)
        self.model_.train(tf.name)
        os.remove(tf.name)

        return self

    def predict(self, X):
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".arff", dir="/dev/shm", delete=False)
        to_arff(X, None, self.n_classes_, tf)
        tf.close()

        pred = np.zeros(len(X), dtype=np.int32)

        for i, r in enumerate(self.model_.predict(tf.name)):
            pred[i] = int(r.predicted[5])

        os.remove(tf.name)

        return self.classes_[pred]


# OK3 =========================================================================

from sklearn.preprocessing import LabelBinarizer

import sys
sys.path.append("./ok3")
import ok3

# Note: memory should be freed through ok3.close()


class OK3RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10,
                       max_features="auto",
                       random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        # Check params
        self.n_features_ = X.shape[1]
        random_state = check_random_state(self.random_state)

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

        self.params_ = ok3.ok3config()
        self.params_.init_rf(k=max_features)
        self.params_.nbtrees = self.n_estimators
        self.params_.randomseed = random_state.randint(1000000)
        self.params_.returnimportances = False
        self.params_.returntrees = False
        self.params_.savepred = 1
        self.params_.verbose = 0

        # Convert data
        self.lb_ = LabelBinarizer().fit(y)
        y = self.lb_.transform(y)

        # Run
        self.X_ = np.ascontiguousarray(X, dtype=np.float32)
        self.y_ = np.ascontiguousarray(y, dtype=np.float32)
        self.ls_ = np.arange(len(self.X_), dtype=np.int32)
        w = np.array([], dtype=np.float32)

        ok3.learn(self.X_, self.y_, self.ls_, w, self.params_)

        return self

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)

        pred = ok3.predict(X, self.params_)

        if len(self.lb_.classes_) > 2:
            pred = np.argmax(pred, axis=1)
        else:
            pred = (pred[:, 0] > 0.5).astype(np.int32)

        return self.lb_.classes_[pred]


class OK3ExtraTreesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10,
                       max_features="auto",
                       random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        # Check params
        self.n_features_ = X.shape[1]
        random_state = check_random_state(self.random_state)

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

        self.params_ = ok3.ok3config()
        self.params_.init_extratrees(k=max_features)
        self.params_.nbtrees = self.n_estimators
        self.params_.randomseed = random_state.randint(1000000)
        self.params_.returnimportances = False
        self.params_.returntrees = False
        self.params_.savepred = 1
        self.params_.verbose = 0

        # Convert data
        self.lb_ = LabelBinarizer().fit(y)
        y = self.lb_.transform(y)

        # Run
        self.X_ = np.ascontiguousarray(X, dtype=np.float32)
        self.y_ = np.ascontiguousarray(y, dtype=np.float32)
        self.ls_ = np.arange(len(self.X_), dtype=np.int32)
        w = np.array([], dtype=np.float32)

        ok3.learn(self.X_, self.y_, self.ls_, w, self.params_)

        return self

    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)

        pred = ok3.predict(X, self.params_)

        if len(self.lb_.classes_) > 2:
            pred = np.argmax(pred, axis=1)
        else:
            pred = (pred[:, 0] > 0.5).astype(np.int32)

        return self.lb_.classes_[pred]


# Orange ======================================================================

import Orange
import orange
import orngEnsemble


def make_orange_dataset(X, y, n_classes):
    classes = [str(c) for c in range(n_classes)]
    columns = ["feature_%d" % i for i in range(X.shape[1])]
    input_vars = map(orange.FloatVariable, tuple(columns))
    class_var = orange.EnumVariable("y", values=classes)
    domain = orange.Domain(input_vars, class_var)
    examples = np.hstack((X, y.reshape(-1,1)))
    return orange.ExampleTable(domain, examples)

class OrangeRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10,
                       max_features="auto",
                       random_state=None):
        self.n_estimators = n_estimators
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

        # Convert data
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y = np.searchsorted(self.classes_, y)
        X = X.astype(np.float32)

        self.table_ = make_orange_dataset(X, y, self.n_classes_)

        # Run
        self.model_ = orngEnsemble.RandomForestLearner(self.table_,
                                                       trees=self.n_estimators,
                                                       attributes=max_features)

        return self

    def predict(self, X):
        X = X.astype(np.float32)
        pred = np.zeros(len(X))

        for i in range(len(X)):
            instance = Orange.data.Instance(self.table_.domain, X[i, :].tolist()+[0])
            pred[i] = self.model_(instance)

        pred = pred.astype(np.int32)

        return self.classes_[pred]
