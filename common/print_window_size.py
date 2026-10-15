import pandas
import collections
import matplotlib.pyplot as plt

from common.params import *


def print_window_size(window_sizes, name):
    number_of_candidates_counted = collections.Counter(sorted(window_sizes))
    df = pandas.DataFrame.from_dict(number_of_candidates_counted, orient="index")
    ax = df.plot(kind="bar", legend=False)
    ax.set_title("Window Size")
    ax.set_xlabel("Candidate Count")
    ax.set_ylabel("Frequency")
    plt.savefig(f"{ROOT_PATH}/candidates_bar/candidates_bar_{name}.png")
    return
