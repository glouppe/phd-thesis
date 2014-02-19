import numpy as np
import matplotlib.pyplot as plt

np.random.seed(777)

blue = (0, 0, 1.0)
red = (1.0, 0, 0)
gray = (0.7, 0.7, 0.7)

n_samples = 20

X = np.empty(n_samples)
X[:n_samples / 2] = np.sort(np.random.normal(loc=-1.0, size=n_samples / 2))
X[n_samples / 2:] = np.sort(np.random.normal(loc=1.0, size=n_samples / 2))
y = np.zeros(n_samples)
y[n_samples / 2:] = 1

plt.plot([-3,3], [0,0], '-', color='k')

X_ = np.sort(X)
for i in range(len(X_) - 1):
    s = (X[i] + X[i+1]) / 2.0
    plt.plot([s,s], [0.00001, -0.00001], '-', color=gray)

plt.scatter(X[:n_samples / 2], np.zeros(n_samples / 2), color=blue)
plt.scatter(X[n_samples / 2:], np.zeros(n_samples / 2), color=red)

s1 = (X[4] + X[5]) / 2.0
s2 = (X[5] + X[6]) / 2.0

plt.plot([s1,s1], [0.00003, -0.00003], '-', color='k')
plt.text(s1, 0.000035, "$v$", fontsize=15, horizontalalignment='center')
plt.text((s1+(-3)) / 2.0, -0.00003, "$t_L (X_j \leq v)$", fontsize=15, horizontalalignment='center')
plt.text((s1+3) / 2.0, -0.00003, "$t_R (X_j > v)$", fontsize=15, horizontalalignment='center')

#plt.plot([s2,s2], [0.00003, -0.00003], '-', color='k')

print s1, s2

plt.text(3, -0.000005, "$X_j$", fontsize=15)

plt.show()

