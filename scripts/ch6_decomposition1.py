import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

from demo import make_led
from ID3 import RandomizedID3Classifier, RandomizedID3Ensemble

n_trees = 3000

X, y = make_led()
fig, axs = plt.subplots(1, 2)

ax = axs[0]
clf = RandomizedID3Ensemble(n_estimators=n_trees,
                            base_estimator=RandomizedID3Classifier(k=1)).fit(X, y)
imp = clf.feature_importances_
ax.imshow(imp, cmap=cm.gist_heat_r, interpolation="nearest", vmin=0, vmax=0.4)
ax.set_yticklabels(["$X_%d$" % (i) for i in range(X.shape[1]+1)])
ax.set_title("$K=1$")

ax = axs[1]
clf = RandomizedID3Ensemble(n_estimators=n_trees,
                            base_estimator=RandomizedID3Classifier(k=X.shape[1])).fit(X, y)
imp = clf.feature_importances_
img = ax.imshow(imp, cmap=cm.gist_heat_r, interpolation="nearest", vmin=0, vmax=0.4)
ax.set_yticklabels(["$X_%d$" % (i) for i in range(X.shape[1]+1)])
ax.set_title("$K=%d$" % X.shape[1])

cax, kw = matplotlib.colorbar.make_axes([ax for ax in axs.flat])
cb = plt.colorbar(img, cax=cax, **kw)
cb.set_ticks([0, 0.2, 0.4])

plt.show()

