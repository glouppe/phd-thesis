import numpy as np
import glob
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import brewer2mpl

cmap_curve = [(1.0, 0, 0), (0, 0, 1.0)]
cmap_bar = brewer2mpl.get_map('RdYlGn', 'diverging', 9).mpl_colors

layout = {
    "RandomForestClassifier": {"name": "Scikit-Learn-RF", "order": 0},
    "RandomForestRegressor": {"name": "Scikit-Learn-RF", "order": 0},
    "ExtraTreesClassifier": {"name": "Scikit-Learn-ETs", "order": 1},
    "ExtraTreesRegressor": {"name": "Scikit-Learn-ETs", "order": 1},
    "OpenCV": {"name": "OpenCV-RF", "order": 2},
    "OpenCV-ETs": {"name": "OpenCV-ETs", "order": 3},
    "OK3-RandomForest": {"name": "OK3-RF", "order": 4},
    "OK3-ExtraTrees": {"name": "OK3-ETs", "order": 5},
    "R-randomForest": {"name": "R-RF", "order": 7},
    "Weka": {"name": "Weka-RF", "order": 6},
    "Orange": {"name": "Orange-RF", "order": 8},
}


def get(data, field, first=True):
    d = data
    for token in field.split("__"):
        d = d[token]

    if isinstance(d, list) and first:
        return d[0]
    else:
        return d


def groupby(filenames, group_fields, param_field, stat_field):
    all_data = {}

    for filename in filenames:
        with open(filename, "r") as fd:
            data = json.load(fd)

            key = []
            for field in group_fields:
                key.append(get(data, field))
            key = tuple(key)

            if key not in all_data:
                all_data[key] = []

            all_data[key].append((get(data, param_field), get(data, stat_field, first=False)))

    for key in all_data:
        all_data[key] = sorted(all_data[key])

    return all_data


def plot_curve(all_data, x_label=None, y_label=None, width=0.2, curve=True, filename=None):
    matplotlib.rc("font", size=13)
    title = all_data.keys()[0][1]
    title = title.split(".")[0]

    all_data = sorted([(layout[key[0]]["order"], layout[key[0]]["name"], all_data[key]) for key in all_data])
    offset = len(all_data) * width + width/2.0

    fig, ax = plt.subplots()

    for i, (key, name, data) in enumerate(all_data):
        xticks = [t[0] for t in data]
        x = [offset*t[0]+i*width for t in data]
        y = [np.mean(t[1]) for t in data]

        if x_label == "n_estimators":
            xticks = [t[0] for t in data if t[0] < 1000]
            x = [offset*t[0]+i*width for t in data if t[0] < 1000]
            y = [np.mean(t[1]) for t in data if t[0] < 1000]

        if y_label == "MSE":
            y = [-y_i for y_i in y]

        if curve:
            ax.plot(xticks, y, label=name, color=cmap_curve[i])
        else:
            ax.bar(x, y, width=width, label=name, color=cmap_curve[i])

    if curve:
        ax.set_xlim(xticks[0], xticks[-1])
    else:
        ax.set_xlim(-2*width+x[0], x[-1]+2*width)
        ax.set_xticks(x)
        ax.set_xticklabels(xticks)

    if x_label is not None: ax.set_xlabel(x_label)
    if y_label is not None: ax.set_ylabel(y_label)

    ax.set_title(title)
    ax.legend(loc="best")

    if filename:
        plt.savefig("%s.pdf" % filename)
        plt.savefig("%s.jpg" % filename)
        plt.close("all")
    else:
        plt.show()


def plot_bar(all_data, y_label=None, width=0.2, filename=None):
    title = all_data.keys()[0][1]
    title = title.split(".")[0]

    all_data = sorted([(layout[key[0]]["order"], layout[key[0]]["name"], all_data[key]) for key in all_data])
    fig, ax = plt.subplots()

    for i, (key, name, data) in enumerate(all_data):
        y_mean = np.mean(data[0][1])
        rects = ax.bar([i*width], [y_mean], width=width, label=name, color=cmap_bar[key])
        rect = rects[0]
        plt.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height(), '%.2f' % y_mean, ha='center', va='bottom', fontsize=9)

    if y_label is not None: ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_xlim(-width, len(layout)*width-width)
    ax.legend(loc="best", prop={"size": 9})

    if filename:
        plt.savefig("%s.pdf" % filename)
        plt.savefig("%s.jpg" % filename)
        plt.close("all")
    else:
        plt.show()


def make_5_4_1():
    # Plot result on artifical data
    regression = ["make_friedman1", "make_friedman2", "make_friedman3"]
    classification = ["make_hastie_10_2", "make_waveforms", "make_twonorm", "make_threenorm", "make_ringnorm"]

    params = [("n_estimators", "params__n_estimators", True),
              ("max_features", "params__max_features", False),
              ("bootstrap", "params__bootstrap", False),
              ("n_train", "stats__n_train", True),
              ("n_features", "stats__n_features", True)]

    stats = [("time_fit", "Fit time (s)"),
             ("time_predict", "Predict time(s)"),
             ("score_make_scorer(accuracy_score)", "Accuracy"),
             ("score_make_scorer(roc_auc_score, needs_threshold=True)", "AUC"),
             ("score_make_scorer(mean_squared_error, greater_is_better=False)", "MSE"),
             ("score_make_scorer(r2_score)", "R2"),
             ("leaves", "Leaves"),
             ("average_depth", "Average depth")]

    for dataset in regression+classification:
        for prefix, param_field, curve in params:
            files = [f for f in glob.glob("output/%s_*_%s*" % (prefix, dataset))]

            if len(files) == 0:
                continue

            for stat_field, label in stats:
                print dataset, prefix, stat_field

                try:
                    plot_curve(groupby(files, ["estimator", "generator"], param_field, "stats__%s" % stat_field),
                               x_label=prefix,
                               y_label=label,
                               filename="figs/generators/%s/%s_%s" % (dataset, prefix, stat_field),
                               curve=curve)
                except:
                    print "Failed!"

def make_5_4_2_plots():
    # Plot results on datasets
    datasets = ["diabetes.npz", "dig44.npz", "ionosphere.npz", "pendigits.npz",
                "letter.npz", "liver.npz", "musk2.npz", "ring-norm.npz", "satellite.npz",
                "segment.npz", "sonar.npz", "spambase.npz", "two-norm.npz", "vehicle.npz",
                "vowel.npz", "waveform.npz", "cifar10.npz", "mnist3vs8.npz", "mnist4vs9.npz", "mnist.npz",
                "isolet.npz", "arcene.npz", "breast2.npz", "madelon.npz", "marti0.npz",
                "reged0.npz", "secom.npz", "tis.npz", "sido0.npz"]

    for dataset in datasets:
        print dataset
        files = glob.glob("output/default_*_%s*" % dataset)
        plot_bar(groupby(files, ["estimator", "generator"], "estimator", "stats__time_fit"), y_label="Fit time (s)", filename="figs/datasets/%s_fit" % dataset)
        plot_bar(groupby(files, ["estimator", "generator"], "estimator", "stats__time_predict"), y_label="Predict time (s)", filename="figs/datasets/%s_predict" % dataset)
        plot_bar(groupby(files, ["estimator", "generator"], "estimator", "stats__score_make_scorer(accuracy_score)"), y_label="Accuracy", filename="figs/datasets/%s_accuracy" % dataset)


def make_5_4_2_table():
    impls = ["RandomForestClassifier",
             "ExtraTreesClassifier",
             "OpenCV",
             "OpenCV-ETs",
             "OK3-RandomForest",
             "OK3-ExtraTrees",
             "Weka",
             "R-randomForest",
             "Orange"]

    datasets = ["diabetes.npz", "dig44.npz", "ionosphere.npz", "pendigits.npz",
                "letter.npz", "liver.npz", "musk2.npz", "ring-norm.npz", "satellite.npz",
                "segment.npz", "sonar.npz", "spambase.npz", "two-norm.npz", "vehicle.npz",
                "vowel.npz", "waveform.npz", "cifar10.npz", "mnist3vs8.npz", "mnist4vs9.npz", "mnist.npz",
                "isolet.npz", "arcene.npz", "breast2.npz", "madelon.npz", "marti0.npz",
                "reged0.npz", "secom.npz", "tis.npz", "sido0.npz"]

    all_stats = {}

    for dataset in datasets:
        all_stats[dataset] = {}
        files = glob.glob("output/default_*_%s*" % dataset)
        data = groupby(files, ["estimator", "generator"], "estimator", "stats__time_predict")

        for (estimator, _), s in data.items():
            all_stats[dataset][estimator] = np.mean(s[0][1])

    table = np.zeros((len(datasets), len(impls)))

    for i, (dataset, stats) in enumerate(sorted(all_stats.items())):
        for j, impl in enumerate(impls):
            if impls[j] in stats:
                table[i, j] = stats[impls[j]]
            else:
                table[i, j] = np.inf

    speedups = np.zeros(table.shape)

    for i, dataset in enumerate(sorted(datasets)):
        for j, impl in enumerate(impls):
            speedups[i, j] = table[i, j] / table[i, 0]

    speedups = np.ma.masked_array(speedups, np.isinf(speedups))

    print "\\begin{tabular}{|c|",
    for j, impl in enumerate(impls):
        print "c",
    print "|}"
    print "\\hline"

    for j, impl in enumerate(impls):
        print "&", layout[impl]["name"],
    print "\\\\"
    print "\\hline"
    print "\\hline"

    for i, dataset in enumerate(sorted(datasets)):
        print "\\textsc{%s}" % dataset.split(".")[0],
        min_j = np.argmin(speedups[i])

        for j, impl in enumerate(impls):
            if j == min_j:
                print "& \\textbf{%.2f}" % speedups[i, j],
            else:
                print "& %.2f" % speedups[i, j],
        print "\\\\"
    print "\\hline"
    print "\\hline"

    print "\\textit{Average}",
    means = speedups.mean(axis=0)
    min_j = np.argmin(means)
    for j, m in enumerate(means):
        if j == min_j:
            print "& \\textbf{%.2f}" % m,
        else:
            print "& %.2f" % m,
    print "\\\\"
    print "\\textit{Median}",
    medians = np.ma.median(speedups, axis=0)
    min_j = np.argmin(medians)
    for j, m in enumerate(medians):
        if j == min_j:
            print "& \\textbf{%.2f}" % m,
        else:
            print "& %.2f" % m,
    print "\\\\"
    print "\\hline"
    print "\\end{tabular}"

if __name__ == "__main__":
    # make_5_4_1()
    make_5_4_2_plots()
    #make_5_4_2_table()

