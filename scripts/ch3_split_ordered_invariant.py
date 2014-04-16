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

indices = np.argsort(X)
X_ = X[indices]
for i in range(len(X_) - 1):
    s = (X_[i]+X_[i+1]) / 2.0
    plt.plot([s,s], [0.00001, 0], ':', color=gray)

plt.plot([-3,-3], [0.00001, 0], '-', color='k')

y = y[indices]

def gini(p):
    p1 = 1.0 * np.sum(p) / len(p)
    p0 = 1.0 - p1
    return p0 * (1 - p0) + p1 * (1 - p1)

s = []
delta = []

for i in range(1, 8):
    i_t = gini(y)
    i_t_L = gini(y[:i])
    i_t_R = gini(y[i:])
    p_L = 1.0 * i / n_samples
    p_R = 1.0 - p_L

    s.append((X_[i-1] + X_[i]) / 2.0)
    delta.append(i_t - p_L * i_t_L - p_R * i_t_R)

delta = np.array(delta)
delta /= np.max(delta)
delta *= 0.00001

plt.plot(s, delta, "o-")


plt.scatter(X[:n_samples / 2], np.zeros(n_samples / 2), color=blue)
plt.scatter(X[n_samples / 2:], np.zeros(n_samples / 2), color=red)

s1 = X_[6]
s2 = X_[7]
smid = (s1+s2) / 2.0

#plt.plot([s2,s2], [0.00001, -0.00001], '-', color=gray)
plt.text(-3, 0.0000105, "$\Delta i(s_j^v, t)$", fontsize=15, horizontalalignment='center')
plt.text(s1, 0.00000095, "$x_{i-1,j}$", fontsize=15, horizontalalignment='center')
plt.text(s2, 0.00000095, "$x_{i,j}$", fontsize=15, horizontalalignment='center')
plt.text(smid, 0.0000105, "$v^\prime_k$", fontsize=15, horizontalalignment='center')
plt.text((smid+(-3)) / 2.0, -0.000001, "${\cal L}^{v^\prime_k}_{t_L}$", fontsize=15, horizontalalignment='center')
plt.text((smid+3) / 2.0, -0.000001, "${\cal L}^{v^\prime_k}_{t_R}$", fontsize=15, horizontalalignment='center')

plt.annotate(
    '', xy=(-3, -0.0000015), xycoords = 'data',
    xytext = (smid, -0.0000015), textcoords = 'data',
    arrowprops = {'arrowstyle':'<->'})
plt.annotate(
    '', xy=(smid, -0.0000015), xycoords = 'data',
    xytext = (3, -0.0000015), textcoords = 'data',
    arrowprops = {'arrowstyle':'<->'})

plt.annotate("$\Delta$", xy=(s[3], delta[3]), xycoords='data', xytext=(s[3]-0.5, delta[3]-0.000001), textcoords='data', arrowprops={'arrowstyle':'->'})

plt.text(3, -0.0000007, "$X_j$", fontsize=15)

plt.show()

