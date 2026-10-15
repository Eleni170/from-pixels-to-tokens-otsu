import numpy
# import scipy
import random
import skimage
import matplotlib.pyplot as plt


def create_otsu_pixels():
    number_of_intensities = 256
    number_of_pixels = 64
    x = [random.randint(0, number_of_intensities - 1) for _ in range(number_of_pixels)]
    frequencies, bins = numpy.histogram(x, bins=number_of_intensities, range=(0, number_of_intensities))
    otsu_threshold = skimage.filters.threshold_otsu(numpy.array(x))

    # x_array = numpy.array(x)
    # kde = scipy.stats.gaussian_kde(x_array)
    # x_range = numpy.linspace(0, number_of_intensities - 1, 1000)
    # kde_values = kde(x_range)
    # kde_values *= len(x) * (number_of_intensities / number_of_pixels)

    plt.figure(figsize=(10, 6))
    plt.xlim(0, number_of_intensities)
    plt.ylim(0, frequencies.max())

    plt.plot(bins[:-1], frequencies, marker="o", linestyle="None", color="blue")
    # plt.plot(x_range, kde_values, color="blue", label="Kernel Density Estimation of Pixel Intensities")
    plt.axvline(x=otsu_threshold, color="red", linestyle="--", label=f"Otsu's Threshold: {otsu_threshold}")

    plt.title("Sample Grayscale Image with Otsu's Threshold")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Pixel Frequency")

    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig("otsu_pixels.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    create_otsu_pixels()
