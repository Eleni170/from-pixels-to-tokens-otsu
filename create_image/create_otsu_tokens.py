import numpy
import skimage
import collections
import matplotlib.pyplot as plt

from create_image.variable_x import variable_x


def create_otsu_tokens():
    token_probabilities = collections.Counter(variable_x)
    sorted_items = sorted(token_probabilities.items())
    labels, values = zip(*sorted_items)
    labels, values = numpy.array(labels), numpy.array(values)

    reconstructed_data = numpy.repeat(labels, values)
    otsu_threshold = skimage.filters.threshold_otsu(reconstructed_data)

    low_prob_mask = labels < otsu_threshold
    low_prob_labels = labels[low_prob_mask]
    low_prob_values = values[low_prob_mask]

    bin_edges = numpy.linspace(0, otsu_threshold, 6)
    bin_counts, _ = numpy.histogram(
        low_prob_labels, bins=bin_edges, weights=low_prob_values
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(10, 6))
    plt.xlim(0, 1.0)
    plt.ylim(0, 600)

    high_prob_mask = labels >= otsu_threshold

    plt.scatter(labels[high_prob_mask], values[high_prob_mask], s=10, color="grey")
    plt.scatter(bin_centers, bin_counts, s=10, color="grey")

    plt.axvline(x=otsu_threshold, color="red", linestyle="--", label=f"Otsu's Threshold: %.4f" % otsu_threshold)

    candidate_tokens = ["republic", "presidential"]

    for i, value in enumerate(candidate_tokens):
        plt.text(labels[-(i + 1)], values[-(i + 1)], value, rotation=30)

    # "constitutional", "representative", "democratic", "republican"
    candidate_tokens = ["", "", "democratic", "republican"]
    for i, value in enumerate(candidate_tokens):
        plt.text(bin_centers[-(i + 1)], 1, value, rotation=30)

    plt.title("Histogram of a Sample Generation Step with Otsu's Threshold", fontsize=16)
    plt.xlabel("Token Probability (0-1)", fontsize=12)
    plt.ylabel("Frequency (Number of Tokens)", fontsize=12)

    plt.grid(axis="y", which="both", linestyle="-", alpha=0.5)
    plt.legend(loc="upper right")
    plt.savefig("otsu_tokens.png", dpi=100)
    plt.show()

    return


if __name__ == "__main__":
    create_otsu_tokens()
