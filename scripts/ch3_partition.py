import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(300, 2)
y = (X[:,0] < 0.7) & (X[:,1] > 0.5)

# randomly flips some labels
mask = np.random.permutation(len(X))[:15]
y[mask] = ~y[mask]

X_c1 = X[y == 0]
plt.scatter(X_c1[:, 0], X_c1[:, 1], color=(1.0, 0, 0))

X_c2 = X[y == 1]
plt.scatter(X_c2[:, 0], X_c2[:, 1], color=(0, 0, 1.0))

# draw lines + text
plt.plot([0, 1], [0, 0], color='k', linestyle='-', linewidth=1)
plt.plot([0, 0], [0, 1], color='k', linestyle='-', linewidth=1)
plt.plot([1, 1], [0, 1], color='k', linestyle='-', linewidth=1)
plt.plot([0, 1], [1, 1], color='k', linestyle='-', linewidth=1)

plt.plot([0.7, 0.7], [0, 1.0], color='k', linestyle='-', linewidth=1)
plt.plot([0, 0.7], [0.5, 0.5], color='k', linestyle='-', linewidth=1)

plt.text(0.95, 0.93, r"$t_2$", fontsize=15)
plt.text(0.65, 0.43, r"$t_3$", fontsize=15)
plt.text(0.65, 0.93, r"$t_4$", fontsize=15)

plt.text(0.7, -0.07, "$0.7$", fontsize=15, horizontalalignment='center')
plt.text(-0.07, 0.5, "$0.5$", fontsize=15, verticalalignment='center')

plt.text(1.0, -0.07, "$X_1$", fontsize=15, horizontalalignment='center')
plt.text(-0.07, 1.0, "$X_2$", fontsize=15, verticalalignment='center')
plt.show()

