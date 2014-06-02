import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
cmap = brewer2mpl.get_map('RdYlGn', 'diverging', 7).mpl_colors
#cmap = [(0, 0, 1.0), (1.0, 0, 0)]

all_importances = np.array(
[[0.414, 0.362, 0.327, 0.309, 0.304, 0.305, 0.306],
 [0.583, 0.663, 0.715, 0.757, 0.787, 0.801, 0.799],
 [0.532, 0.512, 0.496, 0.489, 0.483, 0.475, 0.475],
 [0.543, 0.525, 0.484, 0.445, 0.414, 0.409, 0.412],
 [0.658, 0.731, 0.778, 0.810, 0.827, 0.831, 0.835],
 [0.221, 0.140, 0.126, 0.122, 0.122, 0.121, 0.120],
 [0.368, 0.385, 0.392, 0.387, 0.382, 0.375, 0.372]])

n_features = all_importances.shape[0]
for m in range(n_features):
    plt.plot(range(1, n_features+1), all_importances[m, :], "o-", label="X%d" % (m+1), color=cmap[m])

plt.legend(loc="best")
plt.show()

