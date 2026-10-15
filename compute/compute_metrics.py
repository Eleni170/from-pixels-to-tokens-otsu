import nltk
import time
import numpy
import scipy
import evaluate
import collections
import transformers

from common.params import *
from common.compute_f1s_qa import compute_f1s_qa
from common.get_mini_check_results import get_mini_check_results
from common.get_align_score_results import get_align_score_results

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

hallucination_model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "vectara/hallucination_evaluation_model",
    trust_remote_code=True
)


def compute_metrics(tokenizer, model, dataset_dict, name, test):
    claims = []
    contexts = []
    predictions = []
    references = []

    num_distinct_unigrams = []
    num_distinct_bigrams = []
    num_distinct_trigrams = []
    num_distinct_tetragrams = []

    records_certainties = []
    records_certainties_min = []
    records_certainties_max = []
    records_entropies = []
    records_entropies_max = []
    records_entropies_last_token = []
    records_confidence_margins = []
    records_window_sizes = []

    summaries = {}

    file_path = (
        f"{ROOT_PATH}/summaries/"
        f"{model.__class__.__name__}_"
        f"{name}_"
        f"{model.config.name_or_path.replace('/', '-')}_"
        f"{model.generation_config.do_sample}_"
        f"{model.generation_config.num_beams}_"
        f"{model.generation_config.top_k}_"
        f"{model.generation_config.top_p}_"
        f"{model.generation_config.epsilon_cutoff}.txt"
    )

    time_start = time.time()

    dataloader = torch.utils.data.DataLoader(dataset_dict["test"], batch_size=BATCH_SIZES[test])

    dataset_index = 0
    for batch in dataloader:
        batch_contexts = batch["contexts"].to(DEVICE)
        batch_input_ids = batch["input_ids"].to(DEVICE)
        batch_attention_mask = batch["attention_mask"].to(DEVICE)
        batch_labels = batch["labels"].to(DEVICE)

        outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        batch_predictions = [
            sequence[len(batch_input_ids[sequence_index]):]
            for sequence_index, sequence in enumerate(outputs.sequences)
        ]

        predictions += tokenizer.batch_decode(batch_predictions, skip_special_tokens=True)
        references += tokenizer.batch_decode(batch_labels, skip_special_tokens=True)

        if "questions" in name:
            contexts += [
                " ".join([
                    tokenizer.decode(context, skip_special_tokens=True),
                    tokenizer.decode(reference, skip_special_tokens=True)
                ])
                for context, reference in zip(batch_contexts, batch_labels)
            ]
            claims += [
                " ".join([
                    tokenizer.decode(context, skip_special_tokens=True),
                    tokenizer.decode(prediction, skip_special_tokens=True)
                ])
                for context, prediction in zip(batch_contexts, batch_predictions)
            ]
        else:
            contexts += tokenizer.batch_decode(batch_contexts, skip_special_tokens=True)
            claims += tokenizer.batch_decode(batch_predictions, skip_special_tokens=True)

        transition_scores_all = model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            outputs.beam_indices if model.generation_config.num_beams > 1 else None,
            normalize_logits=True
        )

        for batch_index in range(len(batch_input_ids)):
            tokens = outputs.sequences[batch_index][len(batch_input_ids[batch_index]):]
            mask = tokens != tokenizer.pad_token_id
            tokens = tokens[mask]

            unigrams = nltk.ngrams(tokens, 1)
            unigram_counts = collections.Counter(unigrams)
            num_distinct_unigrams.append(len(unigram_counts) / len(tokens))

            bigrams = nltk.ngrams(tokens, 2)
            bigram_counts = collections.Counter(bigrams)
            num_distinct_bigrams.append(len(bigram_counts) / len(tokens))

            trigrams = nltk.ngrams(tokens, 3)
            trigram_counts = collections.Counter(trigrams)
            num_distinct_trigrams.append(len(trigram_counts) / len(tokens))

            tetragrams = nltk.ngrams(tokens, 4)
            tetragram_counts = collections.Counter(tetragrams)
            num_distinct_tetragrams.append(len(tetragram_counts) / len(tokens))

            scores = tuple(score[batch_index] for m, score in zip(mask, outputs.scores) if m)

            summary = tokenizer.decode(batch_predictions[batch_index], skip_special_tokens=True)

            record_window_sizes = [(score != -float("inf")).sum().item() for score in scores]

            transition_scores = transition_scores_all[batch_index][mask]
            record_certainties = [torch.exp(score).item() for score in transition_scores]

            record_entropies = []
            record_confidence_margins = []

            for output_index in range(transition_scores.shape[0]):
                probs = torch.nn.functional.softmax(scores[output_index], dim=-1).cpu().numpy()
                entropy = scipy.stats.entropy(probs)
                record_entropies.append(entropy)

                probs_sorted = numpy.sort(probs)
                confidence_margin = probs_sorted[-1] - probs_sorted[-2]
                record_confidence_margins.append(confidence_margin)

            average_entropy_score = numpy.average(record_entropies)

            records_certainties.append(numpy.average(record_certainties))
            records_certainties_min.append(min(record_certainties))
            records_certainties_max.append(max(record_certainties))

            records_entropies.append(average_entropy_score)
            records_entropies_max.append(max(record_entropies))
            records_entropies_last_token.append(record_entropies[-1])

            records_confidence_margins.append(numpy.average(record_confidence_margins))

            records_window_sizes.append(numpy.average(record_window_sizes))

            summaries[dataset_index] = {
                "certainty": average_entropy_score,
                "completed": summary.strip().endswith("."),
                "length": transition_scores.shape[0],
                "text": summary,
                "window": records_window_sizes[-1]
            }

            dataset_index += 1

    # Compute metrics
    rouge_results = collections.defaultdict(list)
    for pred, ref in zip(predictions, references):
        for k, v in rouge.compute(predictions=[pred], references=[ref]).items():
            rouge_results[k].append(v)
    bert_score_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    f1s_qa = compute_f1s_qa(predictions=predictions, references=references)
    hhem_results = [
        hallucination_model.predict([(pred, ref)]).item()
        for pred, ref in zip(predictions, references)
    ]
    align_score_results = get_align_score_results(contexts, claims)
    mini_check_results = get_mini_check_results(contexts, claims)

    summaries_sorted = sorted(summaries.keys(), key=lambda i: summaries[i]["certainty"], reverse=True)

    if CONDITIONING:
        conditioned_records = sorted(summaries_sorted[int(0.05 * len(summaries_sorted)):])
    else:
        conditioned_records = sorted(summaries_sorted)

    file_summaries_model = open(file_path, "w", encoding="utf-8")
    for summary_sorted_index in summaries_sorted:
        summary_sorted = summaries[summary_sorted_index]["text"]
        file_summaries_model.write(f"-----\n{summary_sorted_index}\n{summary_sorted}\n-----\n")
    file_summaries_model.close()

    average_uncertain_summary = numpy.average([summaries[i]['length'] for i in summaries_sorted[:NUM_SUBSET_RECORDS]])
    average_certain_summary = numpy.average([summaries[i]['length'] for i in summaries_sorted[-NUM_SUBSET_RECORDS:]])
    average_summary = numpy.average([s['length'] for s in summaries.values()])
    print(f"Average summary length (uncertain): %.2f" % average_uncertain_summary)
    print(f"Average summary length (certain): %.2f" % average_certain_summary)
    print(f"Average summary length (all): %.2f" % average_summary)

    distribution_completed = collections.Counter(s['completed'] for s in summaries.values())
    print(f"Distribution of completed/not completed summaries: {distribution_completed}")

    execution_time = (time.time() - time_start) / 60
    print(f"Computation of metrics took %.2f minutes." % execution_time)

    return {
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "rougeLsum": rouge_results["rougeLsum"],
        "bert_precision": bert_score_results["precision"],
        "bert_recall": bert_score_results["recall"],
        "bert_f1": bert_score_results["f1"],
        "f1_qa": f1s_qa,
        "hhem": hhem_results,
        "align_score": align_score_results,
        "mini_check": mini_check_results,
        "num_distinct_unigrams": num_distinct_unigrams,
        "num_distinct_bigrams": num_distinct_bigrams,
        "num_distinct_trigrams": num_distinct_trigrams,
        "num_distinct_tetragrams": num_distinct_tetragrams,
        "conditioned_records": conditioned_records,
        "length": [s["length"] for s in summaries.values()],
        "token_certainty": records_certainties,
        "token_certainty_min": records_certainties_min,
        "token_certainty_max": records_certainties_max,
        "token_entropy": records_entropies,
        "token_entropy_max": records_entropies_max,
        "token_entropy_last_token": records_entropies_last_token,
        "token_confidence_margin": records_confidence_margins,
        "token_window_size": records_window_sizes
    }
