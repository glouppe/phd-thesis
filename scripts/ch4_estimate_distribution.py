import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

blue = (0, 0, 1.0)
green = (0, 0.8, 0)
red = (1.0, 0, 0)
red_alpha = (1.0, 0, 0, 0.1)
gray = (0.7, 0.7, 0.7)

x = np.arange(0, 1, 0.0001)
p_y = norm.pdf(x, 0.6, 0.1)

plt.plot(x, p_y, color=red)
plt.plot([0.5,0.5], [0.0, np.max(p_y)], '-', color=gray)
plt.plot([0.6,0.6], [0.0, np.max(p_y)+0.01], ':', color=gray)
plt.text(0.6, np.max(p_y) + 0.2, r"$\mathbb{E}_{\cal L} \{ p_{\cal L}(Y=\varphi_B(x)) \}$", fontsize=15, horizontalalignment='center')
plt.text(0.6, 1.7, r"$Var_{\cal L}\{ p_{\cal L}(Y=\varphi_B(x)) \}$", fontsize=15, horizontalalignment='left')
plt.annotate(
    '', xy=(0.45, 2.0), xycoords = 'data',
    xytext = (0.75, 2.0), textcoords = 'data',
    arrowprops = {'arrowstyle':'<->'})
plt.annotate(r"$P_{\cal L}(\varphi_{\cal L}(x)\neq \varphi_B(x))$", xy=(0.475, 1.0), xycoords='data', fontsize=15, xytext=(0.2, 1.7), textcoords='data', arrowprops={'arrowstyle':'->'})

plt.fill_between(x, p_y, y2=0, where=x<0.5, color=red_alpha)

plt.ylabel("$P$")
plt.ylim((0., 4.5))
plt.xlim((0., 1.0))
plt.xticks([0.0, 0.5, 1.0])
plt.yticks([])

plt.show()
