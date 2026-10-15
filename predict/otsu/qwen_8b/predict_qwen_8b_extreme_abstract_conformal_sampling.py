import transformers

from common.params import *
from conformal_nucleus_sampling.calibrate import calibrate
from predict.otsu.predict_model_dataset_otsu import predict_model_dataset_otsu
from preprocess.preprocess_natural_instructions import preprocess_natural_instructions
from conformal_nucleus_sampling.generation.conformal_nucleus_sampling_model_for_causal_lm import \
    ConformalNucleusSamplingModelForCausalLM


def predict_qwen_8b_extreme_abstract_conformal_sampling():
    tokenizer = transformers.AutoTokenizer.from_pretrained(QWEN_8B)
    model = transformers.AutoModelForCausalLM.from_pretrained(QWEN_8B, dtype="auto", device_map="auto")

    model.generation_config.do_sample = True
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS_EXTREME_ABSTRACT
    model.generation_config.min_new_tokens = MIN_NEW_TOKENS_EXTREME_ABSTRACT
    model.generation_config.temperature = TEMPERATURE
    model.generation_config.top_k = 0
    model.generation_config.top_p = TOP_P
    model.generation_config.repetition_penalty = REPETITION_PENALTY

    extreme_abstract_dataset_dict = preprocess_natural_instructions(
        tokenizer,
        "task668_extreme_abstract_summarization"
    )

    entropies_quantiles = calibrate(tokenizer, model, extreme_abstract_dataset_dict)

    model = ConformalNucleusSamplingModelForCausalLM.from_pretrained(
        QWEN_8B,
        entropies_quantiles=entropies_quantiles,
        dtype="auto",
        device_map="auto"
    )

    model.generation_config.do_sample = True
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS_EXTREME_ABSTRACT
    model.generation_config.min_new_tokens = MIN_NEW_TOKENS_EXTREME_ABSTRACT
    model.generation_config.temperature = TEMPERATURE
    model.generation_config.top_k = 0
    model.generation_config.top_p = 1.0
    model.generation_config.repetition_penalty = REPETITION_PENALTY

    predict_model_dataset_otsu(
        tokenizer, model, extreme_abstract_dataset_dict, "qwen8_extreme_abstract_conformal_sampling"
    )
    return


if __name__ == "__main__":
    predict_qwen_8b_extreme_abstract_conformal_sampling()
