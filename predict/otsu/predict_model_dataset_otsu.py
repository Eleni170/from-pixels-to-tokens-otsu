import numpy
import scipy

from common.params import *
from compute.compute_metrics import compute_metrics
from common.print_window_size import print_window_size
from compute.compute_creativity_diversity_metrics import compute_creativity_diversity_metrics


def predict_model_dataset_otsu(tokenizer, model, dataset_dict, name):
    test = "test_otsu"

    results = compute_metrics(tokenizer, model, dataset_dict, name, test)
    results_creativity_diversity = compute_creativity_diversity_metrics(tokenizer, model, dataset_dict, test)

    conditioned_records = results["conditioned_records"]
    del results["conditioned_records"]

    if CONDITIONING:
        results = {
            key: [value[i] for i in conditioned_records]
            for key, value in results.items()
        }

        results_creativity_diversity = {
            key: [value[i] for i in conditioned_records]
            for key, value in results_creativity_diversity.items()
        }

    print("--Quality--")
    print({
        "rouge1": "%.4f" % numpy.average(results["rouge1"]),
        "rouge2": "%.4f" % numpy.average(results["rouge2"]),
        "rougeL": "%.4f" % numpy.average(results["rougeL"]),
        "rougeLsum": "%.4f" % numpy.average(results["rougeLsum"]),
        "bert_precision_avg": "%.4f" % numpy.average(results["bert_precision"]),
        "bert_recall_avg": "%.4f" % numpy.average(results["bert_recall"]),
        "bert_f1_avg": "%.4f" % numpy.average(results["bert_f1"]),
        "f1_qa_avg": "%.4f" % numpy.average(results["f1_qa"])
    })

    print("--Factuality--")
    print({
        "hhem_avg": "%.4f" % numpy.average(results["hhem"]),
        "align_score_avg": "%.4f" % numpy.average(results["align_score"]),
        "mini_check_avg": "%.4f" % numpy.average(results["mini_check"])
    })

    print("--Uncertainty--")
    print({
        "token_certainty_avg": "%.4f" % numpy.average(results["token_certainty"]),
        "token_certainty_min_avg": "%.4f" % numpy.average(results["token_certainty_min"]),
        "token_certainty_max_avg": "%.4f" % numpy.average(results["token_certainty_max"]),
        "token_confidence_margin_avg": "%.4f" % numpy.average(results["token_confidence_margin"]),
        "token_entropy_avg": "%.4f" % numpy.average(results["token_entropy"]),
        "token_entropy_max_avg": "%.4f" % numpy.average(results["token_entropy_max"]),
        "token_entropy_last_token_avg": "%.4f" % numpy.average(results["token_entropy_last_token"]),
        "semantic_entropy_avg": "%.4f" % numpy.average(results_creativity_diversity["semantic_entropy"]),
        "token_window_size_avg": "%.4f" % numpy.average(results["token_window_size"])
    })

    print("--Diversity--")
    print({
        "diversity_level_avg": "%.4f" % numpy.average(results_creativity_diversity["diversity_level"]),
        "num_distinct_unigrams": "%.4f" % numpy.average(results["num_distinct_unigrams"]),
        "num_distinct_bigrams": "%.4f" % numpy.average(results["num_distinct_bigrams"]),
        "num_distinct_trigrams": "%.4f" % numpy.average(results["num_distinct_trigrams"]),
        "num_distinct_tetragrams": "%.4f" % numpy.average(results["num_distinct_tetragrams"])
    })

    print("--Correlations (r, p)--")
    results_correlations = {
        "ROUGEL–BERT F1": (results["rougeL"], results["bert_f1"]),
        "ROUGEL–F1 QA": (results["rougeL"], results["f1_qa"]),
        "ROUGEL–HHEM": (results["rougeL"], results["hhem"]),
        "ROUGEL–AlignScore": (results["rougeL"], results["align_score"]),
        "ROUGEL–MiniCheck": (results["rougeL"], results["mini_check"]),
        "ROUGEL–Certainty": (results["rougeL"], results["token_certainty"]),
        "ROUGEL–Entropy": (results["rougeL"], results["token_entropy"]),
        "ROUGEL–Semantic Entropy": (results["rougeL"], results_creativity_diversity["semantic_entropy"]),
        "ROUGEL–Diversity Level": (results["rougeL"], results_creativity_diversity["diversity_level"]),
        "ROUGEL–Window Size": (results["rougeL"], results["token_window_size"]),
        "BERT F1–F1 QA": (results["bert_f1"], results["f1_qa"]),
        "BERT F1–HHEM": (results["bert_f1"], results["hhem"]),
        "BERT F1–AlignScore": (results["bert_f1"], results["align_score"]),
        "BERT F1–MiniCheck": (results["bert_f1"], results["mini_check"]),
        "BERT F1–Certainty": (results["bert_f1"], results["token_certainty"]),
        "BERT F1–Entropy": (results["bert_f1"], results["token_entropy"]),
        "BERT F1–Semantic Entropy": (results["bert_f1"], results_creativity_diversity["semantic_entropy"]),
        "BERT F1–Diversity Level": (results["bert_f1"], results_creativity_diversity["diversity_level"]),
        "BERT F1–Window Size": (results["bert_f1"], results["token_window_size"]),
        "F1 QA-HHEM": (results["f1_qa"], results["hhem"]),
        "F1 QA-AlignScore": (results["f1_qa"], results["align_score"]),
        "F1 QA-MiniCheck": (results["f1_qa"], results["mini_check"]),
        "F1 QA-Certainty": (results["f1_qa"], results["token_certainty"]),
        "F1 QA-Entropy": (results["f1_qa"], results["token_entropy"]),
        "F1 QA-Semantic Entropy": (results["f1_qa"], results_creativity_diversity["semantic_entropy"]),
        "F1 QA-Diversity Level": (results["f1_qa"], results_creativity_diversity["diversity_level"]),
        "F1 QA-Window Size": (results["f1_qa"], results["token_window_size"]),
        "HHEM-AlignScore": (results["hhem"], results["align_score"]),
        "HHEM-MiniCheck": (results["hhem"], results["mini_check"]),
        "HHEM-Certainty": (results["hhem"], results["token_certainty"]),
        "HHEM-Entropy": (results["hhem"], results["token_entropy"]),
        "HHEM-Semantic Entropy": (results["hhem"], results_creativity_diversity["semantic_entropy"]),
        "HHEM-Diversity Level": (results["hhem"], results_creativity_diversity["diversity_level"]),
        "HHEM-Window Size": (results["hhem"], results["token_window_size"]),
        "AlignScore-MiniCheck": (results["align_score"], results["mini_check"]),
        "AlignScore-Certainty": (results["align_score"], results["token_certainty"]),
        "AlignScore-Entropy": (results["align_score"], results["token_entropy"]),
        "AlignScore-Semantic Entropy": (results["align_score"], results_creativity_diversity["semantic_entropy"]),
        "AlignScore–Diversity Level": (results["align_score"], results_creativity_diversity["diversity_level"]),
        "AlignScore–Window Size": (results["align_score"], results["token_window_size"]),
        "MiniCheck-Certainty": (results["mini_check"], results["token_certainty"]),
        "MiniCheck-Entropy": (results["mini_check"], results["token_entropy"]),
        "MiniCheck-Semantic Entropy": (results["mini_check"], results_creativity_diversity["semantic_entropy"]),
        "MiniCheck–Diversity Level": (results["mini_check"], results_creativity_diversity["diversity_level"]),
        "MiniCheck–Window Size": (results["mini_check"], results["token_window_size"]),
        "Certainty-Entropy": (results["token_certainty"], results["token_entropy"]),
        "Certainty-Semantic Entropy": (results["token_certainty"], results_creativity_diversity["semantic_entropy"]),
        "Certainty–Diversity Level": (results["token_certainty"], results_creativity_diversity["diversity_level"]),
        "Certainty–Window Size": (results["token_certainty"], results["token_window_size"]),
        "Entropy–Semantic Entropy": (results["token_entropy"], results_creativity_diversity["semantic_entropy"]),
        "Entropy–Diversity Level": (results["token_entropy"], results_creativity_diversity["diversity_level"]),
        "Entropy–Window Size": (results["token_entropy"], results["token_window_size"]),
        "Semantic Entropy–Diversity Level": (
            results_creativity_diversity["semantic_entropy"], results_creativity_diversity["diversity_level"]
        ),
        "Semantic Entropy–Window Size": (
            results_creativity_diversity["semantic_entropy"], results["token_window_size"]
        ),
        "Diversity Level–Window Size": (results_creativity_diversity["diversity_level"], results["token_window_size"])
    }
    for key, (x, y) in results_correlations.items():
        r, p = scipy.stats.pearsonr(x, y)
        results_correlations[key] = (f"{r:.4f}", f"{p:.4f}")
    print(results_correlations)

    with open(f"{ROOT_PATH}/results/results_{name}.txt", "w", encoding="utf-8") as f:
        for results_dict in [results, results_creativity_diversity]:
            for key, value in results_dict.items():
                f.write(f"{key}: {[v.item() if isinstance(v, numpy.floating) else v for v in value]}\n")

    print_window_size(results["token_window_size"], name)
    return
