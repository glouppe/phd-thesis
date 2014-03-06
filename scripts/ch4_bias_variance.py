import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

blue = (0, 0, 1.0)
red = (1.0, 0, 0)
gray = (0.7, 0.7, 0.7)

x = np.arange(-10, 10, 0.0001)
p_y = norm.pdf(x, -3.0, 1)
p_y_hat = norm.pdf(x, 3.0, 1.8)

plt.plot(x, p_y, color=blue)
plt.plot(x, p_y_hat, color=red)

plt.plot([-3,-3], [0.0, np.max(p_y)+0.01], ':', color=gray)
plt.text(-3, np.max(p_y) + 0.02, r"$\varphi_B(x)$", fontsize=15, horizontalalignment='center')

plt.plot([3,3], [0.0, np.max(p_y_hat)+0.01], ':', color=gray)
plt.text(3, np.max(p_y_hat) + 0.02, r"$\mathbb{E}_{\cal L} \{ \varphi_{\cal L}(x) \}$", fontsize=15, horizontalalignment='center')

plt.text(0, 0.11, r"$bias^2(x)$", fontsize=15, horizontalalignment='center')
plt.annotate(
    '', xy=(-3, 0.1), xycoords = 'data',
    xytext = (3, 0.1), textcoords = 'data',
    arrowprops = {'arrowstyle':'<->'})

plt.text(-5.1, 0.21, r"$noise(x)$", fontsize=15, horizontalalignment='right')
plt.annotate(
    '', xy=(-5, 0.2), xycoords = 'data',
    xytext = (-1, 0.2), textcoords = 'data',
    arrowprops = {'arrowstyle':'<->'})

plt.text(5.1, 0.21, r"$var(x)$", fontsize=15, horizontalalignment='left')
plt.annotate(
    '', xy=(6, 0.2), xycoords = 'data',
    xytext = (0, 0.2), textcoords = 'data',
    arrowprops = {'arrowstyle':'<->'})

plt.tick_params(axis="x", which="both", bottom="off", top="off", labelbottom="off")
plt.xlabel("$y$")
plt.ylabel("$P$")

plt.show()
