import numpy
import scipy

from common.params import *


def calibrate(tokenizer, model, dataset_dict):
    generation_step_values = []

    dataloader = torch.utils.data.DataLoader(dataset_dict["val"], batch_size=BATCH_SIZES["val_otsu"])

    for batch in dataloader:
        batch_input_ids = batch["input_ids"].to(DEVICE)
        batch_attention_mask = batch["attention_mask"].to(DEVICE)
        batch_labels = batch["labels"].to(DEVICE)

        # Generate and extract outputs from the articles
        outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

        for batch_index in range(len(batch_input_ids)):
            scores = [score[batch_index] for score in outputs.scores]

            for output_index in range(min(len(batch_labels[batch_index]), len(scores))):
                probs = torch.nn.functional.softmax(scores[output_index], dim=-1).cpu().numpy()
                entropy = scipy.stats.entropy(probs)
                reference_token = batch_labels[batch_index][output_index].item()
                generation_step_values.append((probs, entropy, reference_token))

    generation_step_values_sorted = sorted(generation_step_values, key=lambda x: x[1])

    num_bins = 10
    bin_size = int(len(generation_step_values_sorted) / num_bins)

    bins = [
        generation_step_values_sorted[i * bin_size: (i + 1) * bin_size]
        for i in range(num_bins)
    ]

    entropies_quantiles = []
    for b in bins:
        entropy_threshold = b[-1][1]
        s = [p[p >= p[r] - 1e-9].sum() for p, _, r in b]
        q = numpy.quantile(s, numpy.ceil((len(s) + 1) * TOP_P) / len(s), method="higher")
        entropies_quantiles.append((entropy_threshold, q))

    return entropies_quantiles
