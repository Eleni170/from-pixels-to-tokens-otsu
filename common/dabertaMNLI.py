import transformers

from common.params import *

pipe = transformers.pipeline(
    "text-classification",
    model="tasksource/deberta-base-long-nli",
    device=DEVICE,
    batch_size=16
)


def find_similar_sentences_indices(pair_sentences_a, pair_sentences_b):
    data = []
    for sentence_a, sentence_b in zip(pair_sentences_a, pair_sentences_b):
        data.append({"text": sentence_a, "text_pair": sentence_b})
        data.append({"text": sentence_b, "text_pair": sentence_a})

    results = pipe(data)
    similar_sentences_indices = []

    for i in range(0, len(results), 2):
        r1 = results[i]
        r2 = results[i + 1]
        if r1["label"] == "entailment" and r2["label"] == "entailment":
            similar_sentences_indices.append(i // 2)

    return similar_sentences_indices
