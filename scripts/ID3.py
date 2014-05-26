"""
Understanding variable importances in forests of randomized trees.
Gilles Louppe, Louis Wehenkel, Antonio Sutera and Pierre Geurts
NIPS, Lake Tahoe, United States, 2013
http://orbi.ulg.ac.be/handle/2268/155642

This module implements a simplistic randomized ID3 tree classifier
(`RandomizedID3Classifier`), along with its ensemble counter-part
(`RandomizedID3Ensemble`).

Warning: These classes implement `fit` and  `feature_importances_`, but do not
         provide any `predict` method. They only serve as a proof-of-concept.

Author: Gilles Louppe <g.louppe@gmail.com>
License: BSD 3 clause
"""
import copy
import itertools
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.utils import check_random_state

from demo import entropy

MAX_INT = np.iinfo("i").max


class RandomizedID3Classifier(BaseEstimator, ClassifierMixin):
    """Simplistic implementation of an ID3 randomized tree."""

    def __init__(self, k=1, max_depth=None, random_state=None):
        self.k = k
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree_ = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.random_state_ = check_random_state(self.random_state)

        self.values_ = []
        for i in xrange(self.n_features_):
            self.values_.append(np.unique(X[:, i]))

        self.tree_ = self._partition(X,
                                     np.searchsorted(self.classes_, y),
                                     range(self.n_features_),
                                     X.shape[0])

        return self

    def predict(self, X):
        raise NotImplementedError

    def _partition(self, X, y, variables, n_samples, depth=0):
        rng = self.random_state_

        # Leaf
        if len(variables) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            values = 1. * np.bincount(y, minlength=self.n_classes_) / len(y)
            return (values, len(y))

        # Internal node
        else:
            variables = copy.copy(variables)
            n_variables = len(variables)
            n_node = len(X)

            best = None
            best_score = -np.inf
            best_children = None

            features = (rng.permutation(n_variables))[:min(self.k,
                                                           n_variables)]

            for i in features:
                X_i = variables[i]

                children = []

                for xi in self.values_[X_i]:
                    mask_xi = X[:, X_i] == xi
                    if sum(mask_xi) > 0:
                        children.append((X[mask_xi], y[mask_xi], sum(mask_xi)))

                score = ((1. * n_node / n_samples)  # P(B=b)
                         * (entropy(y) - sum([1. * entropy(c_y) * c_n / n_node
                                              for _, c_y, c_n in children])))

                if score > best_score:
                    best = i
                    best_score = score
                    best_children = children

            X_i = variables.pop(best)

            return (X_i,
                    best_score,
                    [self._partition(c_X,
                                     c_y,
                                     variables,
                                     n_samples,
                                     depth=depth+1) for c_X,
                                                        c_y,
                                                        _ in best_children])

    @property
    def feature_importances_(self):
        def _visit(tree, depth):
            if len(tree) == 2:
                pass

            else:
                imp[tree[0], depth] += tree[1]

                for c in tree[2]:
                    _visit(c, depth+1)

        imp = np.zeros((self.n_features_, self.n_features_))
        _visit(self.tree_, 0)

        return imp


class RandomizedID3Ensemble(BaseEnsemble, ClassifierMixin):
    """Simplistic implementation of an ensemble of ID3 randomized trees."""

    def __init__(self, base_estimator=None, n_estimators=10, max_depth=None, random_state=None):
        super(RandomizedID3Ensemble, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=("max_depth",))

        self.max_depth = max_depth
        self.random_state = random_state

    def _validate_estimator(self):
        super(RandomizedID3Ensemble, self)._validate_estimator(
            default=RandomizedID3Classifier())

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)
        self._validate_estimator()
        self.p = X.shape[1]

        for i in xrange(self.n_estimators):
            tree = self._make_estimator()
            tree.set_params(random_state=random_state.randint(MAX_INT))
            tree.fit(X, y)

        return self

    def predict(self, X):
        raise NotImplementedError

    @property
    def feature_importances_(self):
        importances = np.zeros((self.p, self.p))

        for i, tree in enumerate(self.estimators_):
            importances += tree.feature_importances_

        importances /= self.n_estimators

        return importances
