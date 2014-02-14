import numpy as np
import matplotlib.pyplot as plt

blue = (0, 0, 1.0)
red = (1.0, 0, 0)
gray = (0.7, 0.7, 0.7)

# Criterion

def impurity_error(p1, p2):
    return min(p1, p2)

def impurity_entropy(p1, p2):
    if p1 == 0.0 or p1 == 1.0 or p2 == 0.0 or p2 == 1.0:
        return 0.0
    else:
        return -(p1 * np.log2(p1) + p2 * np.log2(p2))

def impurity_gini(p1, p2):
    return p1 * (1 - p1) + p2 * (1 - p2)

# Split

def p(y_t):
    return 1.0 * y_t / np.sum(y_t)

impurity = impurity_gini
y_t = np.array([2, 8], dtype=np.float)
y_t_L = np.array([1, 5], dtype=np.float)
y_t_R = y_t - y_t_L
p_y_t = p(y_t)
p_y_t_L = p(y_t_L)
p_y_t_R = p(y_t_R)

p_L = y_t_L.sum() / y_t.sum()
p_R = y_t_R.sum() / y_t.sum()

i_t = impurity(*p_y_t)
i_t_L = impurity(*p_y_t_L)
i_t_R = impurity(*p_y_t_R)

print "Delta i(s, t) = i(t) - p_L * i(t_L) - p_R * i (t_R)"
print "              = %f - %f * %f - %f * %f" % (i_t, p_L, i_t_L, p_R, i_t_R)
print "              = %f" % (i_t - p_L * i_t_L - p_R * i_t_R, )


fig = plt.figure()
ax = fig.add_subplot(111)

x = np.linspace(0.0, 1.0, num=300)
# ax.plot(x, map(impurity, x, 1-x), label="entropy", color=blue)
ax.plot(x, map(impurity_error, x, 1-x), label="$i_E(t)$", color=gray)
ax.plot(x, map(impurity_entropy, x, 1-x), label="$i_H(t)$", color=blue)
ax.plot(x, map(impurity_gini, x, 1-x), label="$i_G(t)$", color=red)
ax.legend(loc="best")
plt.show()

ax.plot(p_y_t[0], i_t, marker="o", color=red)
ax.plot(p_y_t_L[0], i_t_L, marker="o", color=red)
ax.plot(p_y_t_R[0], i_t_R, marker="o", color=red)

ax.plot((p_y_t[0], p_y_t[0]), (0, i_t), ":", color=gray)
ax.plot((0, p_y_t[0]), (i_t, i_t), ":", color=gray)
ax.annotate("$i(t)$", xy=(0, i_t), xytext=(0+0.01, i_t), va="center")
ax.annotate("$p(c_1|t)$", xy=(p_y_t[0], 0), xytext=(p_y_t[0], 0+0.025), ha="center")

ax.plot((p_y_t_L[0], p_y_t_L[0]), (0, i_t_L), ":", color=gray)
ax.plot((0, p_y_t_L[0]), (i_t_L, i_t_L), ":", color=gray)
ax.annotate("$i(t_L)$", xy=(0, i_t_L), xytext=(0+0.01, i_t_L), va="center")
ax.annotate("$p(c_1|t_L)$", xy=(p_y_t_L[0], 0), xytext=(p_y_t_L[0], 0+0.025), ha="center")

ax.plot((p_y_t_R[0], p_y_t_R[0]), (0, i_t_R), ":", color=gray)
ax.plot((0, p_y_t_R[0]), (i_t_R, i_t_R), ":", color=gray)
ax.annotate("$i(t_R)$", xy=(0, i_t_R), xytext=(0+0.01, i_t_R), va="center")
ax.annotate("$p(c_1|t_R)$", xy=(p_y_t_R[0], 0), xytext=(p_y_t_R[0], 0+0.025), ha="center")

ax.plot((p_y_t_L[0], p_y_t_R[0]), (i_t_L, i_t_R), "-", color=gray)
ax.plot((p_y_t[0], p_y_t[0]), (i_t, p_L * i_t_L + p_R * i_t_R), "-", color=red)
ax.plot(p_y_t[0], p_L * i_t_L + p_R * i_t_R, marker="o", color=gray)
ax.annotate("$\Delta i(s, t) = %.3f$" % abs(i_t - p_L * i_t_L - p_R * i_t_R), xy=(p_y_t[0], i_t - 0.5*(i_t - p_L * i_t_L - p_R * i_t_R)), xytext=(p_y_t[0]+0.05, i_t - 0.5*(i_t - p_L * i_t_L - p_R * i_t_R)),  arrowprops=dict(arrowstyle="->"), va="center")

#ax.legend(loc="best")
plt.show()
