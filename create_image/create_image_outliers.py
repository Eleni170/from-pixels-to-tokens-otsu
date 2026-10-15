import ast
import numpy
import pandas
import matplotlib.pyplot as plt

from common.params import *


def create_image_outliers():
    outliers_percentages = numpy.arange(0, 55, 5)
    plt.figure(figsize=(10, 6))
    plt.xlim(0, 50)

    method_config = {
        "t_sampling": {"label": "Typical", "color": "dimgray"},
        "top_k_sampling": {"label": "Top-$k$", "color": "green"},
        "top_p_sampling": {"label": "Top-$p$", "color": "red"},
        "e_sampling": {"label": "Epsilon", "color": "yellow"},
        "conformal_sampling": {"label": "Conformal", "color": "orange"},
        "otsu_no_top_k": {"label": "Otsu (all)", "color": "lightblue"},
        "otsu_top_k": {"label": "Otsu ($k$)", "color": "olive"}
    }

    filenames = ["t_sampling", "top_k_sampling", "top_p_sampling", "e_sampling", "conformal_sampling", "otsu_no_top_k",
                 "otsu_top_k"]
    for filename in filenames:
        with open(f"{ROOT_PATH}/results/results_qwen14_amazon_review_{filename}.txt", encoding="utf-8") as file:
            results = {}

            filelines = file.readlines()
            for line in filelines:
                metric, values = line.split(':', 1)
                results[metric] = ast.literal_eval(values.strip().replace("nan", "None"))

            results = pandas.DataFrame(results)
            sorted_results = results.sort_values(by="token_entropy", ascending=False).to_dict(orient="list")
            sorted_conditioned_results = []
            for outliers_percentage in outliers_percentages:
                sorted_conditioned_list = sorted_results["rougeL"][
                    int((outliers_percentage / 100) * len(sorted_results["rougeL"])):
                ]
                sorted_conditioned_results.append(numpy.average(sorted_conditioned_list))

        plt.plot(
            outliers_percentages,
            sorted_conditioned_results,
            marker="o",
            label=method_config[filename]["label"],
            color=method_config[filename]["color"]
        )

    plt.title("ROUGE-L (Amazon Review)", fontsize=16, pad=35, fontweight="bold")
    plt.xlabel("Percentage of Outliers Removed (%)")
    plt.ylabel("Average ROUGE-L")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=7, frameon=True)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("rougel_amazon.png", dpi=100, bbox_inches="tight")
    plt.close()

    return


if __name__ == "__main__":
    create_image_outliers()
