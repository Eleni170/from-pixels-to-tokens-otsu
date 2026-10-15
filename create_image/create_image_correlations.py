import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt


def create_image_correlations():
    model_order = ["Llama-8B", "Llama-70B", "Mistral-24B", "Qwen-8B", "Qwen-14B"]
    decoding_order = ["Typical", "Top-$k$", "Top-$p$", "Epsilon", "Conformal", "Otsu (all)", "Otsu ($k$)"]

    data = {
        "Model": ["Llama-8B"] * 7 + ["Llama-70B"] * 7 + ["Mistral-24B"] * 7 + ["Qwen-8B"] * 7 + ["Qwen-14B"] * 7,
        'Decoding method': ["Typical", "Top-$k$", "Top-$p$", "Epsilon", "Conformal", "Otsu (all)", "Otsu ($k$)"] * 5,
        "Values": [0] * 35
    }

    dataframe = pandas.DataFrame(data)

    dataframe["Model"] = pandas.Categorical(
        dataframe["Model"], categories=model_order, ordered=True
    )
    dataframe["Decoding method"] = pandas.Categorical(
        dataframe["Decoding method"], categories=decoding_order, ordered=True
    )

    heatmap_data = dataframe.pivot(index="Decoding method", columns="Model", values="Values")

    plt.figure(figsize=(10, 8))

    seaborn.heatmap(
        heatmap_data,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation Coefficient (r)"}
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.title(
        "BERT-F1–Certainty (Extreme Abstract)",
        fontsize=16,
        pad=20,
        fontweight="bold"
    )
    plt.savefig("bert_f1_certainty_extreme.png", dpi=100, bbox_inches="tight")
    plt.close()

    return


if __name__ == '__main__':
    create_image_correlations()
