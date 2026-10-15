import nltk
import time
import numpy
import scipy
import itertools
import bert_score
import collections

from common.params import *
from common.dabertaMNLI import find_similar_sentences_indices

similarity_model = bert_score.BERTScorer(model_type="bert-base-uncased")


def compute_creativity_diversity_metrics(tokenizer, model, dataset_dict, test):
    semantic_entropies = []
    num_clusters = []

    diversity_levels = []

    num_generations = 3 if ("card" in test or "mental" in test) else 5

    time_start = time.time()

    dataloader = torch.utils.data.DataLoader(dataset_dict["test"], batch_size=BATCH_SIZES[test])

    for batch in dataloader:
        batch_input_ids = batch["input_ids"].to(DEVICE)
        batch_attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_generations,
            return_dict_in_generate=True,
            output_scores=True
        )

        outputs_summaries = outputs.sequences[:, batch_input_ids.shape[1]:]

        decoded_summaries = tokenizer.batch_decode(outputs_summaries, skip_special_tokens=True)

        transition_scores = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            outputs.beam_indices if model.generation_config.num_beams > 1 else None,
            normalize_logits=True
        ).tolist()

        for batch_index in range(len(decoded_summaries) // num_generations):
            record_summaries = decoded_summaries[batch_index * num_generations:(batch_index + 1) * num_generations]
            record_scores = transition_scores[batch_index * num_generations:(batch_index + 1) * num_generations]
            record_sentences = [nltk.sent_tokenize(summary) for summary in record_summaries]

            summaries_list = tokenizer(record_summaries, add_special_tokens=False)["input_ids"]
            sentences_list = [
                [tokenizer(sentence, add_special_tokens=False)["input_ids"] for sentence in sentences]
                for sentences in record_sentences
            ]

            summaries_combinations = list(itertools.combinations(range(len(record_summaries)), 2))
            pairs = [(record_summaries[a], record_summaries[b]) for a, b in summaries_combinations]
            candidates = [p[0] for p in pairs]
            references = [p[1] for p in pairs]

            _, _, bert_f1 = similarity_model.score(candidates, references)
            diversity_levels.append(numpy.average(bert_f1.tolist()))

            clusters = []

            pair_sentences_a = []
            pair_sentences_b = []
            indices = []
            for a, b in summaries_combinations:
                sentences_a = record_sentences[a]
                sentences_b = record_sentences[b]

                for x in range(len(sentences_a)):
                    for y in range(len(sentences_b)):
                        pair_sentences_a.append(sentences_a[x])
                        pair_sentences_b.append(sentences_b[y])
                        indices.append([a, x, b, y])

            similar_sentences_indices = find_similar_sentences_indices(pair_sentences_a, pair_sentences_b)
            for similar_sentences_index in similar_sentences_indices:
                a, x, b, y = indices[similar_sentences_index]
                clusters.append([(a, x), (b, y)])

            for r in range(len(record_summaries)):
                for s in range(len(record_sentences[r])):
                    if not any((r, s) in cluster for cluster in clusters):
                        clusters.append([(r, s)])

            for x, y in itertools.combinations(clusters, 2):
                if not set(x).isdisjoint(y):
                    x += y
                    try:
                        clusters.remove(y)
                    except ValueError:
                        pass

            clusters = [sorted(set(x)) for x in clusters]
            clusters_scores = []

            for cluster in clusters:
                summary_indices = []
                cluster_scores = []
                for summary_index, sentence_index in cluster:
                    summary_tokens = summaries_list[summary_index]
                    sentence_tokens = sentences_list[summary_index][sentence_index]
                    indices = [i for i in range(len(summary_tokens) - len(sentence_tokens) + 1)
                               if summary_tokens[i:i + len(sentence_tokens)] == sentence_tokens]

                    index = 0
                    while index < len(indices):
                        if (summary_index, index) not in summary_indices:
                            if index + len(sentence_tokens) == len(summaries_list[summary_index]):
                                rc = record_scores[summary_index][index: index + len(sentence_tokens) + 1]
                            else:
                                rc = record_scores[summary_index][index: index + len(sentence_tokens)]

                            cluster_scores.append(sum(rc))
                            summary_indices.append((summary_index, index))
                            break

                        index += 1

                clusters_scores.append(sum(cluster_scores))

            clusters_scores = torch.nn.functional.softmax(
                torch.tensor(clusters_scores, dtype=torch.float),
                dim=0
            ).numpy()

            semantic_entropies.append(scipy.stats.entropy(clusters_scores))
            num_clusters.append(len(clusters))

    print(f"Distribution of number of creativity clusters: {collections.Counter(num_clusters)}")

    execution_time = (time.time() - time_start) / 60
    print(f"Computation of creativity-diversity metrics took %.2f minutes." % execution_time)

    return {
        "semantic_entropy": semantic_entropies,
        "diversity_level": diversity_levels
    }
