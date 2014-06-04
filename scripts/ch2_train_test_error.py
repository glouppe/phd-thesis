from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_friedman1
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


# Compute train/test error curves on Friedman1
def error_curves(estimator, parameter, parameter_values, n_repeat=100):
    all_train_errors = []
    all_test_errors = []

    for i in range(n_repeat):
        X, y = make_friedman1(n_samples=200)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

        train_errors = []
        test_errors = []

        for j, p in enumerate(parameter_values):
            est = estimator(**{parameter: p})
            est.fit(X_train, y_train)

            train_errors.append(mean_squared_error(y_train, est.predict(X_train)))
            test_errors.append(mean_squared_error(y_test, est.predict(X_test)))

        all_train_errors.append(train_errors)
        all_test_errors.append(test_errors)

    return all_train_errors, all_test_errors

parameter_values = np.arange(1, 100, dtype=np.int)
all_train_errors, all_test_errors = error_curves(DecisionTreeRegressor,
                                                 "min_samples_split",
                                                 parameter_values)


# Plot the error curves
all_train_errors = np.array(all_train_errors)
all_test_errors = np.array(all_test_errors)

for i, train_errors in enumerate(all_train_errors):
    plt.plot(parameter_values[::-1], train_errors, color=(0, 0, 1, 0.1))
plt.plot(parameter_values[::-1], np.mean(all_train_errors, axis=0),
         color=(0, 0, 1), label="Training error")

for i, test_errors in enumerate(all_test_errors):
    plt.plot(parameter_values[::-1], test_errors, color=(1, 0, 0, 0.1))
plt.plot(parameter_values[::-1], np.mean(all_test_errors, axis=0),
         color=(1, 0, 0), label="Test error")

m = np.mean(all_test_errors, axis=0)
i = np.argmin(m)
plt.vlines((parameter_values[::-1])[i], 0, 30, color=(0.7, 0.7, 0.7))
plt.ylim([0, 30])

plt.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
plt.xlabel("Model complexity")
plt.ylabel("Mean square error")
plt.legend(loc="best")

plt.show()
