import numpy as np
import matplotlib.pyplot as plt

blue = (0, 0, 1.0)
green = (0, 0.8, 0)
red = (1.0, 0, 0)
red_alpha = (1.0, 0, 0, 0.001)
gray = (0.7, 0.7, 0.7)

# Settings
n_repeat = 100       # Number of iterations for computing expectations
n_train = 30        # Size of the training set
n_test = 1000       # Size of the test set
noise = 0.1**0.5         # Standard deviation of the noise
np.random.seed(0)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

estimators = [("Degree = 1", Pipeline([("polynomial_features", PolynomialFeatures(degree=1, include_bias=False)), ("linear_regression", LinearRegression())])),
              ("Degree = 5", Pipeline([("polynomial_features", PolynomialFeatures(degree=5, include_bias=False)), ("linear_regression", LinearRegression())])),
              ("Degree = 15", Pipeline([("polynomial_features", PolynomialFeatures(degree=15, include_bias=False)), ("linear_regression", LinearRegression())])),]

n_estimators = len(estimators)

# Generate data
def f(x):
    x = x.ravel()

    return np.cos(2.5 * np.pi * x)

def generate(n_samples, noise, n_repeat=1):
    X = np.random.rand(n_samples)
    X = np.sort(X)

    if n_repeat == 1:
        y = f(X) + np.random.normal(0.0, noise, n_samples)
    else:
        y = np.zeros((n_samples, n_repeat))

        for i in range(n_repeat):
            y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)

    X = X.reshape((n_samples, 1))

    return X, y

X_train = []
y_train = []

for i in range(n_repeat):
    X, y = generate(n_samples=n_train, noise=noise)
    X_train.append(X)
    y_train.append(y)

X_test, y_test = generate(n_samples=n_test, noise=noise, n_repeat=n_repeat)

plt.figure(figsize=(14, 8))

# Loop over estimators to compare
for n, (name, estimator) in enumerate(estimators):
    # Compute predictions
    y_predict = np.zeros((n_test, n_repeat))

    for i in xrange(n_repeat):
        estimator.fit(X_train[i], y_train[i])
        y_predict[:, i] = estimator.predict(X_test)

    # Bias^2 + Variance + Noise decomposition of the mean squared error
    y_error = np.zeros(n_test)

    for i in range(n_repeat):
        for j in range(n_repeat):
            y_error += (y_test[:, j] - y_predict[:, i]) ** 2

    y_error /= (n_repeat * n_repeat)

    y_noise = np.var(y_test, axis=1)
    y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
    y_var = np.var(y_predict, axis=1)

    print("{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
          " + {3:.4f} (var) + {4:.4f} (noise)".format(name,
                                                      np.mean(y_error),
                                                      np.mean(y_bias),
                                                      np.mean(y_var),
                                                      np.mean(y_noise)))

    # Plot figures
    ax = plt.subplot(2, n_estimators, n + 1)
    plt.setp(ax, xticks=(), yticks=())
    plt.plot(X_test, f(X_test), color=blue)
    plt.plot(X_train[0], y_train[0], ".b")
    plt.plot(X_test, y_predict[:, 0], color=gray)

    for i in range(1, n_repeat):
        plt.plot(X_test, y_predict[:, i], color=red_alpha, alpha=0.05)

    plt.plot(X_test, np.mean(y_predict, axis=1), color=red,
             label="$\mathbb{E}_{LS} \^y(x)$")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0., 1.0))
    plt.ylim((-2, 2))
    plt.title(name)

    ax = plt.subplot(2, n_estimators, n_estimators + n + 1)
    plt.setp(ax, xticks=(), yticks=())
    plt.plot(X_test, y_error, color=gray, label="$error(x)$")
    plt.plot(X_test, y_bias, color=blue, label="$bias^2(x)$"),
    plt.plot(X_test, y_var, color=red, label="$var(x)$"),
    plt.plot(X_test, y_noise, color=green, label="$noise(x)$")
    plt.xlabel("x")
    plt.xlim((0., 1.0))
    plt.ylim((0, 2.0))

    if n == 0:
        plt.legend(loc="upper left", prop={"size": 11})

plt.show()
