import matplotlib.pyplot as plt
import numpy as np


def parity_plot(truth, prediction, title, stat_str=""):
    plt.scatter(truth, prediction, alpha=0.1)
    plt.xlabel("truth")
    plt.ylabel("prediction")
    min_val = min(np.min(truth), np.min(prediction)) - 0.5
    max_val = max(np.max(truth), np.max(prediction)) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="-")
    plt.plot([min_val, max_val], [min_val + 1, max_val + 1], color="red", linestyle="--", alpha=0.25)
    plt.plot([min_val, max_val], [min_val - 1, max_val - 1], color="red", linestyle="--", alpha=0.25)
    plt.ylim(min_val, max_val)
    plt.xlim(min_val, max_val)
    plt.text(min_val, max_val - 0.1, stat_str, horizontalalignment="left", verticalalignment="top")
    plt.title(title)
    return plt.gcf()
