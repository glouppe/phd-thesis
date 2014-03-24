import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

blue = (0, 0, 1.0)
green = (0, 0.8, 0)
red = (1.0, 0, 0)
red_alpha = (1.0, 0, 0, 0.001)
gray = (0.7, 0.7, 0.7)

results = [[],[],
["RandomForestRegressor-K=1",3.527128,2.820386,0.706743,0.063868,0.009973,0.286104,0.420639],
["RandomForestRegressor-K=2",3.036291,2.333874,0.702417,0.075537,0.011347,0.314841,0.387576],
["RandomForestRegressor-K=3",2.823907,2.109897,0.714009,0.087809,0.012335,0.349486,0.364523],
["RandomForestRegressor-K=4",2.715613,1.979086,0.736527,0.102472,0.014302,0.391750,0.344778],
["RandomForestRegressor-K=5",2.643232,1.887080,0.756151,0.111790,0.015411,0.421380,0.334772],
["RandomForestRegressor-K=6",2.642354,1.851498,0.790856,0.125342,0.016268,0.466556,0.324300],
["RandomForestRegressor-K=7",2.636296,1.822316,0.813980,0.134200,0.017159,0.495746,0.318234],
["RandomForestRegressor-K=8",2.623646,1.784344,0.839303,0.146081,0.018631,0.531100,0.308202],
["RandomForestRegressor-K=9",2.645439,1.780447,0.864992,0.152977,0.019492,0.558601,0.306390],
["RandomForestRegressor-K=10",2.638901,1.753437,0.885464,0.160371,0.020184,0.583494,0.301970],
["ExtraTreesRegressor-K=1",3.376099,2.723586,0.652514,0.051864,0.009532,0.230752,0.421761],
["ExtraTreesRegressor-K=2",2.801100,2.146534,0.654566,0.060858,0.011926,0.258086,0.396480],
["ExtraTreesRegressor-K=3",2.536644,1.886837,0.649807,0.067322,0.012756,0.273424,0.376383],
["ExtraTreesRegressor-K=4",2.409943,1.745583,0.664360,0.076519,0.016511,0.302962,0.361399],
["ExtraTreesRegressor-K=5",2.330165,1.651706,0.678459,0.086137,0.017063,0.331515,0.346944],
["ExtraTreesRegressor-K=6",2.285386,1.597063,0.688323,0.092147,0.019216,0.349667,0.338655],
["ExtraTreesRegressor-K=7",2.263983,1.553772,0.710211,0.100322,0.020510,0.378116,0.332094],
["ExtraTreesRegressor-K=8",2.246997,1.528167,0.718831,0.107167,0.021703,0.396323,0.322507],
["ExtraTreesRegressor-K=9",2.236845,1.495768,0.741077,0.115699,0.023020,0.423894,0.317183],
["ExtraTreesRegressor-K=10",2.232862,1.469781,0.763081,0.123849,0.024420,0.451778,0.311304]]

max_features = range(1, 10+1)

ax = plt.subplot(1, 2, 1)
plt.plot(max_features, [results[1+k][1] for k in max_features], 'o-', color=blue, label='Random Forest')
plt.plot(max_features, [results[1+k][2] for k in max_features], 'o--', color=blue)
plt.plot(max_features, [results[1+k][3] for k in max_features], 'o:', color=blue)
plt.plot(max_features, [results[11+k][1] for k in max_features], 'o-', color=red, label='Extremely Randomized Trees')
plt.plot(max_features, [results[11+k][2] for k in max_features], 'o--', color=red)
plt.plot(max_features, [results[11+k][3] for k in max_features], 'o:', color=red)
plt.legend(loc="best")
plt.xlabel("$K$")

plt.subplot(1, 2, 2, sharex=ax)
plt.plot(max_features, [results[1+k][4] for k in max_features], 'o-', color=blue)
plt.plot(max_features, [results[11+k][4] for k in max_features], 'o-', color=red)
plt.xlabel("$K$")
plt.ylabel("$\\rho$")

plt.show()
