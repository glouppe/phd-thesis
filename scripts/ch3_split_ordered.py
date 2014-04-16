import numpy as np
import matplotlib.pyplot as plt

np.random.seed(54)

blue = (0, 0, 1.0)
red = (1.0, 0, 0)
gray = (0.7, 0.7, 0.7)

n_samples = 10

X = np.empty(n_samples)
X[:n_samples / 2] = np.sort(np.random.normal(loc=-1.0, size=n_samples / 2))
X[n_samples / 2:] = np.sort(np.random.normal(loc=1.0, size=n_samples / 2))
y = np.zeros(n_samples)
y[n_samples / 2:] = 1

plt.plot([-3,3], [0,0], '-', color='k')

X_ = np.sort(X)
for i in range(len(X_) - 1):
    s = X[i]
    plt.plot([s,s], [0.00001/2, 0], '-', color=gray)

plt.plot([-3,-3], [0.00001, 0], '-', color=gray)

plt.scatter(X[:n_samples / 2], np.zeros(n_samples / 2), color=blue)
plt.scatter(X[n_samples / 2:], np.zeros(n_samples / 2), color=red)

s1 = X_[6]
s2 = X_[7]
smid = (s1+s2) / 2.0

plt.plot([s1,s1], [0.00001/2, 0], '-', color='k')
plt.text(s1, 0.0000055, "$v_k$", fontsize=15, horizontalalignment='center')
plt.text(s2, 0.0000055, "$v_{k+1}$", fontsize=15, horizontalalignment='center')
plt.text(smid, 0.000001, "$v^\prime_k$", fontsize=15, horizontalalignment='center')
plt.plot([smid,smid], [0.00001/2, 0], ':', color=gray)
plt.text((s1+(-3)) / 2.0,  -0.000001, "${\cal L}^{v_k}_{t_L}$", fontsize=15, horizontalalignment='center')
plt.text((s1+3) / 2.0,  -0.000001, "${\cal L}^{v_k}_{t_R}$", fontsize=15, horizontalalignment='center')

plt.annotate(
    '', xy=(-3, -0.0000015), xycoords = 'data',
    xytext = (s1, -0.0000015), textcoords = 'data',
    arrowprops = {'arrowstyle':'<->'})
plt.annotate(
    '', xy=(s1, -0.0000015), xycoords = 'data',
    xytext = (3, -0.0000015), textcoords = 'data',
    arrowprops = {'arrowstyle':'<->'})

#plt.plot([s2,s2], [0.00003, -0.00003], '-', color='k')


plt.text(3, -0.0000007, "$X_j$", fontsize=15)

plt.show()

