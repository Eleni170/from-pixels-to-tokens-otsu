import scipy

from common.params import *
from transformers.generation.logits_process import LogitsProcessor


class ConformalNucleusSamplingLogitsProcessor(LogitsProcessor):
    def __init__(self, entropies_quantiles, filter_value: float = -float("Inf")):
        self.entropies_quantiles = entropies_quantiles
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape

        for i in range(batch_size):
            row = scores[i]
            valid_mask = row != self.filter_value
            filtered_row = row[valid_mask]

            sorted_logits, sorted_indices = torch.sort(filtered_row, descending=False)
            probs = sorted_logits.softmax(dim=-1)
            cumulative_probs = probs.cumsum(dim=-1)

            entropy = scipy.stats.entropy(probs.cpu().numpy())

            if entropy > self.entropies_quantiles[-1][0]:
                q = self.entropies_quantiles[-1][1]
            else:
                index = 0
                while index < len(self.entropies_quantiles):
                    entropy_threshold = self.entropies_quantiles[index][0]
                    if entropy <= entropy_threshold:
                        q = self.entropies_quantiles[index][1]
                        break
                    index += 1

            mask = torch.zeros(vocab_size, dtype=torch.bool, device=scores.device)
            mask[valid_mask] = torch.as_tensor(
                cumulative_probs <= (1 - q), dtype=torch.bool, device=scores.device
            )

            scores[i] = scores[i].masked_fill(mask, self.filter_value)

        return scores
