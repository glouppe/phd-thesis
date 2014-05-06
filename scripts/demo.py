"""
Understanding variable importances in forests of randomized trees.
Gilles Louppe, Louis Wehenkel, Antonio Sutera and Pierre Geurts
NIPS, Lake Tahoe, United States, 2013
http://orbi.ulg.ac.be/handle/2268/155642

This demo reproduces Table 2 from the paper. It also shows that using Extra-
Trees from Scikit-Learn, or an ensemble of randomized ID3 trees (see ID3.py)
give identical results.

Figure 2 from the paper can be obtained using the 2d array importances values
yielded by a `RandomizedID3Ensemble` (see the commented code at the bottom).

Author: Gilles Louppe <g.louppe@gmail.com>
License: BSD 3 clause
"""
import itertools
import numpy as np

from sklearn.utils import check_random_state


# Datasets ====================================================================

def make_led(irrelevant=0):
    """Generate exhaustively all samples from the 7-segment problem.

    Parameters
    ----------
    irrelevant : int, optional (default=0)
        The number of irrelevant binary features to add. Since samples are
        generated exhaustively, this makes the size of the resulting dataset
        2^(irrelevant) times larger.

    Returns
    -------
    X, y
    """
    data = np.array([[0, 0, 1, 0, 0, 1, 0, 1],
                     [1, 0, 1, 1, 1, 0, 1, 2],
                     [1, 0, 1, 1, 0, 1, 1, 3],
                     [0, 1, 1, 1, 0, 1, 0, 4],
                     [1, 1, 0, 1, 0, 1, 1, 5],
                     [1, 1, 0, 1, 1, 1, 1, 6],
                     [1, 0, 1, 0, 0, 1, 0, 7],
                     [1, 1, 1, 1, 1, 1, 1, 8],
                     [1, 1, 1, 1, 0, 1, 1, 9],
                     [1, 1, 1, 0, 1, 1, 1, 0]])

    X, y = np.array(data[:, :7], dtype=np.bool), data[:, 7]

    if irrelevant > 0:
        X_ = []
        y_ = []

        for i in xrange(10):
            for s in itertools.product(range(2), repeat=irrelevant):
                X_.append(np.concatenate((X[i], s)))
                y_.append(i)

        X = np.array(X_, dtype=np.bool)
        y = np.array(y_)

    return X, y


def make_led_sample(n_samples=200, irrelevant=0, random_state=None):
    """Generate random samples from the 7-segment problem.

    Parameters
    ----------
    n_samples : int, optional (default=200)
        The number of samples to generate.

    irrelevant : int, optional (default=0)
        The number of irrelevant binary features to add.

    Returns
    -------
    X, y
    """

    random_state = check_random_state(random_state)

    data = np.array([[0, 0, 1, 0, 0, 1, 0, 1],
                     [1, 0, 1, 1, 1, 0, 1, 2],
                     [1, 0, 1, 1, 0, 1, 1, 3],
                     [0, 1, 1, 1, 0, 1, 0, 4],
                     [1, 1, 0, 1, 0, 1, 1, 5],
                     [1, 1, 0, 1, 1, 1, 1, 6],
                     [1, 0, 1, 0, 0, 1, 0, 7],
                     [1, 1, 1, 1, 1, 1, 1, 8],
                     [1, 1, 1, 1, 0, 1, 1, 9],
                     [1, 1, 1, 0, 1, 1, 1, 0]])

    data = data[random_state.randint(0, 10, n_samples)]
    X, y = np.array(data[:, :7],  dtype=np.bool), data[:, 7]

    if irrelevant > 0:
        X = np.hstack((X, random_state.rand(n_samples, irrelevant) > 0.5))

    return X, y


# Formulae ====================================================================

from gmpy import comb

def binomial(k, n):
    """Return the number of combinations of k elements among a collection of
       size n."""
    if k < 0:
        return 0
    elif k > n:
        return 0
    else:
        return comb(int(n), int(k))


def entropy(X):
    """Return the entropy (in base 2) of a discrete variable X, encoded as a
       1d array."""
    e = 0.
    n_samples = len(X)

    for count in np.bincount(X):
        p = 1. * count / n_samples

        if p > 0:
            e -= p * np.log2(p)

    return e

def mdi_importance(X_m, X, y):
    """The MDI importance of X_m for Y, as computed with an infinite ensemble
       of fully developed totally randomized trees.

    This is a direct implementation of Equation 3 from the paper.

    Parameters
    ----------
    X_m : int
        The variable for which the importance is computed. It corresponds
        to the column in X (from 0 to p-1).

    X : array of shape (N, p)
        The input data (X_0, X_1, ... X_{p-1}). X should be large enough
        to accurately represent the actual data distribution.

    y : array of shape (N,)
        The Y variable.

    Returns
    -------
    imp : array of size (p,)
        The decomposition of the importance of X_m along its degree of
        interaction with the other input variables, i.e the p outter terms
        in Equation 3. The actual importance Imp(X_m) amounts np.sum(imp).
    """
    n_samples, p = X.shape

    variables = range(p)
    variables.pop(X_m)
    imp = np.zeros(p)

    values = []
    for i in xrange(p):
        values.append(np.unique(X[:, i]))

    for k in xrange(p):
        # Weight of each B of size k
        coef = 1. / (binomial(k, p) * (p - k))

        # For all B of size k
        for B in itertools.combinations(variables, k):
            # For all values B=b
            for b in itertools.product(*[values[B[j]] for j in xrange(k)]):
                mask_b = np.ones(n_samples, dtype=np.bool)

                for j in xrange(k):
                    mask_b &= X[:, B[j]] == b[j]

                X_, y_ = X[mask_b, :], y[mask_b]
                n_samples_b = len(X_)

                if n_samples_b > 0:
                    children = []

                    for xi in values[X_m]:
                        mask_xi = X_[:, X_m] == xi
                        children.append(y_[mask_xi])

                    imp[k] += (coef
                               * (1. * n_samples_b / n_samples)  # P(B=b)
                               * (entropy(y_) -
                                  sum([entropy(c) * len(c) / n_samples_b
                                       for c in children])))

    return imp


# Demo ========================================================================

if __name__ == "__main__":
    # Generate data
    n_trees = 5000

    X, y = make_led()
    p = X.shape[1]

    results = np.empty((p, p + 1))

    # Theoretical values
    for i in range(p):
        results[i, 0] = sum(mdi_importance(i, X, y))

    # Empirical results
    for i in range(p):
        # Using scikit-learn
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=n_trees,
                                   max_features=i + 1,
                                   criterion="entropy",
                                   n_jobs=-1).fit(X, y)

        # Note: Variable importances in Scikit-Learn are normalized by
        #       default. Use normalize=False to disable normalization.

        results[:, i + 1] = sum(tree.tree_.compute_feature_importances(normalize=False)
                                for tree in clf.estimators_) / clf.n_estimators

        # # Using a simplistic (but slower) randomized ID3 tree classifier
        # from ID3 import RandomizedID3Classifier, RandomizedID3Ensemble
        # clf = RandomizedID3Ensemble(n_estimators=n_trees,
        #                             base_estimator=RandomizedID3Classifier(k=i + 1)).fit(X, y)

        # # Note: Here clf.feature_importances is a 2d array of shape (p, p).
        # #       In particular, it could be used to regenerate Figure 2 from
        # #       the paper.

        # results[:, i + 1] = np.sum(clf.feature_importances_, axis=1)


    # Print results
    print "Table 2:"
    print "Variable importances as computed with an ensemble of randomized " \
          "trees, for increasing values of $K$. Importances at $K=1$ follow " \
          "their theoretical values, as predicted by Equation 3 in Theorem 1. " \
          "However, as $K$ increases, importances diverge due to masking " \
          "effects. In accordance with Theorem 2, their sum is also always " \
          "equal to $I(X_{1}, ..., X_{7}; Y) = H(Y) = log2(10)= 3.321$ " \
          "since inputs allow to perfectly predict the output."
    print

    print "\tEqn.3",
    for m in range(p):
        print "\tK=%d" % (m + 1),
    print

    for m in range(p):
        print "X_%d" % (m + 1),
        for j in range(p + 1):
            print "\t%.4f" % results[m, j],
        print

    print "Sum",
    for j in range(p + 1):
        print "\t%.4f" % sum(results[:, j]),
