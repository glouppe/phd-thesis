import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

blue = (0, 0, 1.0)
green = (0, 0.8, 0)
red = (1.0, 0, 0)
red_alpha = (1.0, 0, 0, 0.001)
gray = (0.7, 0.7, 0.7)



results = [["PERTRegressor",3.574108,2.890852,0.683256,0.056576,0.008611,0.256336,0.426920],
["Bagging",3.139157,2.126251,1.012905,0.167573,0.019077,0.678590,0.334315],
["RandomForestRegressor-K=1",3.714658,2.983417,0.731241,0.065858,0.008813,0.301901,0.429340],
["RandomForestRegressor-K=2",3.368340,2.625463,0.742877,0.077040,0.010500,0.337024,0.405854],
["RandomForestRegressor-K=3",3.193635,2.430771,0.762864,0.086496,0.012148,0.370005,0.392860],
["RandomForestRegressor-K=4",3.118717,2.312676,0.806041,0.101932,0.013640,0.427690,0.378351],
["RandomForestRegressor-K=5",3.085201,2.249191,0.836011,0.113693,0.014243,0.469263,0.366748],
["RandomForestRegressor-K=6",3.085762,2.208252,0.877510,0.126165,0.015005,0.518625,0.358885],
["RandomForestRegressor-K=7",3.080685,2.173961,0.906723,0.136779,0.015962,0.556302,0.350422],
["RandomForestRegressor-K=8",3.102127,2.154872,0.947256,0.147335,0.016376,0.600895,0.346360],
["RandomForestRegressor-K=9",3.126621,2.145447,0.981174,0.158014,0.017572,0.641499,0.339675],
["RandomForestRegressor-K=10",3.139157,2.126251,1.012905,0.167573,0.019077,0.678590,0.334315],
["ExtraTreesRegressor-K=1",3.502027,2.816081,0.685946,0.057377,0.009345,0.259587,0.426358],
["ExtraTreesRegressor-K=2",3.055530,2.359293,0.696237,0.065768,0.011442,0.288088,0.408149],
["ExtraTreesRegressor-K=3",2.871624,2.153474,0.718150,0.076130,0.013269,0.325073,0.393077],
["ExtraTreesRegressor-K=4",2.755848,2.014910,0.740938,0.085197,0.014199,0.358499,0.382439],
["ExtraTreesRegressor-K=5",2.698942,1.937569,0.761374,0.093898,0.015927,0.389243,0.372130],
["ExtraTreesRegressor-K=6",2.670654,1.890338,0.780316,0.102993,0.017499,0.419299,0.361017],
["ExtraTreesRegressor-K=7",2.653183,1.844617,0.808566,0.111418,0.018889,0.452685,0.355882],
["ExtraTreesRegressor-K=8",2.646671,1.812010,0.834661,0.120494,0.020072,0.485768,0.348893],
["ExtraTreesRegressor-K=9",2.628936,1.780231,0.848705,0.126758,0.022476,0.506421,0.342284],
["ExtraTreesRegressor-K=10",2.640260,1.767285,0.872976,0.135874,0.023241,0.537841,0.335135]]



max_features = range(1, 10+1)

ax = plt.subplot(1, 2, 1)

# plt.plot(max_features, [results[0][1] for k in max_features], '--', color=green, label='PERT (MSE)')
# plt.plot(max_features, [results[1][1] for k in max_features], '-.', color=green, label='Bagging (MSE)')

plt.plot(max_features, [results[1+k][1] for k in max_features], 'o-', color=blue, label='Random Forest')
plt.plot(max_features, [results[1+k][2] for k in max_features], 'o--', color=blue)
plt.plot(max_features, [results[1+k][3] for k in max_features], 'o:', color=blue)

plt.plot(max_features, [results[11+k][1] for k in max_features], 'o-', color=red, label='Extremely Randomized Trees')
plt.plot(max_features, [results[11+k][2] for k in max_features], 'o--', color=red)
plt.plot(max_features, [results[11+k][3] for k in max_features], 'o:', color=red)
plt.legend(loc="best")
plt.xlabel("$K$")

plt.subplot(1, 2, 2, sharex=ax)

# plt.plot(max_features, [results[0][4] for k in max_features], '--', color=green, label='PERT ($rho$)')
# plt.plot(max_features, [results[1][4] for k in max_features], '-.', color=green, label='Bagging ($rho$)')
plt.plot(max_features, [results[1+k][4] for k in max_features], 'o-', color=blue)
plt.plot(max_features, [results[11+k][4] for k in max_features], 'o-', color=red)
plt.xlabel("$K$")
plt.ylabel("$\\rho$")
plt.show()
