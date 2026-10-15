import torch
import skimage

from transformers.generation.logits_process import LogitsProcessor


class OtsuThresholdingLogitsProcessor(LogitsProcessor):
    def __init__(self, filter_value: float = -float("Inf")):
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape

        for i in range(batch_size):
            row = scores[i]
            valid_mask = row != self.filter_value
            filtered_row = row[valid_mask]
            filtered_row = torch.nn.functional.softmax(filtered_row, dim=-1)

            otsu_threshold = skimage.filters.threshold_otsu(filtered_row.detach().cpu().numpy())

            mask = torch.zeros(vocab_size, dtype=torch.bool, device=scores.device)
            mask[valid_mask] = torch.as_tensor(
                filtered_row < otsu_threshold, dtype=torch.bool, device=scores.device
            )

            scores[i] = scores[i].masked_fill(mask, self.filter_value)

        return scores
