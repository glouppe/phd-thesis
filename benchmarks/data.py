import numpy as np

from sklearn.utils import check_random_state


def make_waveforms(n_samples=300, random_state=None):
    """Make the waveforms dataset. (CART)"""
    random_state = check_random_state(random_state)

    def h1(x):
        if x < 7:
            return x
        elif x < 13:
            return 13.-x
        else:
            return 0.

    def h2(x):
        if x < 9:
            return 0.
        elif x < 15:
            return x-9.
        else:
            return 21.-x

    def h3(x):
        if x < 5:
            return 0.
        elif x < 11:
            return x-5.
        elif x < 17:
            return 17.-x
        else:
            return 0.

    u = random_state.rand(n_samples)
    y = random_state.randint(low=0, high=3, size=n_samples)
    X = random_state.normal(size=(n_samples, 21))

    for i in range(n_samples):
        if y[i] == 0:
            ha = h1
            hb = h2
        elif y[i] == 1:
            ha = h1
            hb = h3
        else:
            ha = h2
            hb = h3

        for m in np.arange(1, 21+1):
            X[i, m-1] += u[i] * ha(m) + (1 - u[i]) * hb(m)

    return X, y


def make_ringnorm(n_samples=300, random_state=None):
    """Make the ring-norm dataset. (Breiman, Tech. report 460.)"""
    random_state = check_random_state(random_state)
    a = 1. / 20.**0.5

    y = random_state.randint(low=0, high=2, size=n_samples)
    X = np.zeros((n_samples, 20))

    negatives = (y == 0)
    positives = (y == 1)

    X[negatives] = random_state.multivariate_normal(mean=np.zeros(20), cov=4.*np.eye(20), size=negatives.sum())
    X[positives] = random_state.normal(loc=[a]*20, size=(positives.sum(), 20))

    return X, y


def make_twonorm(n_samples=300, random_state=None):
    """Make the two-norm dataset. (Breiman, Tech. report 460.)"""
    random_state = check_random_state(random_state)
    a = 2. / 20.**0.5

    y = random_state.randint(low=0, high=2, size=n_samples)
    X = np.zeros((n_samples, 20))

    negatives = (y == 0)
    positives = (y == 1)

    X[negatives] = random_state.normal(loc=[a]*20, size=(negatives.sum(), 20))
    X[positives] = random_state.normal(loc=[-a]*20, size=(positives.sum(), 20))

    return X, y

def make_threenorm(n_samples=300, random_state=None):
    """Make the three-norm dataset. (Breiman, Tech. report 460.)"""
    random_state = check_random_state(random_state)
    a = 2. / 20.**0.5

    y = random_state.randint(low=0, high=4, size=n_samples)
    X = np.zeros((n_samples, 20))

    class0 = (y == 0)
    class1 = (y == 1)
    class2 = (y >= 2)

    X[class0] = random_state.normal(loc=[a]*20, size=(class0.sum(), 20))
    X[class1] = random_state.normal(loc=[-a]*20, size=(class1.sum(), 20))
    X[class2] = random_state.normal(loc=[a,-a]*10, size=(class2.sum(), 20))

    y[class0] = 0
    y[class1] = 0
    y[class2] = 1

    return X, y
