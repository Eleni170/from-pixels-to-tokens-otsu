import transformers

from common.params import *
from conformal_nucleus_sampling.calibrate import calibrate
from predict.otsu.predict_model_dataset_otsu import predict_model_dataset_otsu
from preprocess.preprocess_natural_instructions import preprocess_natural_instructions
from conformal_nucleus_sampling.generation.conformal_nucleus_sampling_model_for_causal_lm import \
    ConformalNucleusSamplingModelForCausalLM


def predict_llama_8b_counterfactual_conformal_sampling():
    tokenizer = transformers.AutoTokenizer.from_pretrained(LLAMA_8B)
    tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
    model = transformers.AutoModelForCausalLM.from_pretrained(LLAMA_8B, dtype="auto", device_map="auto")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    model.generation_config.do_sample = True
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS_COUNTERFACTUAL
    model.generation_config.min_new_tokens = MIN_NEW_TOKENS_COUNTERFACTUAL
    model.generation_config.temperature = TEMPERATURE
    model.generation_config.top_k = 0
    model.generation_config.top_p = TOP_P
    model.generation_config.repetition_penalty = REPETITION_PENALTY

    counterfactual_dataset_dict = preprocess_natural_instructions(
        tokenizer,
        "task269_csrg_counterfactual_story_generation"
    )

    entropies_quantiles = calibrate(tokenizer, model, counterfactual_dataset_dict)

    model = ConformalNucleusSamplingModelForCausalLM.from_pretrained(
        LLAMA_8B,
        entropies_quantiles=entropies_quantiles,
        dtype="auto",
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    model.generation_config.do_sample = True
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS_COUNTERFACTUAL
    model.generation_config.min_new_tokens = MIN_NEW_TOKENS_COUNTERFACTUAL
    model.generation_config.temperature = TEMPERATURE
    model.generation_config.top_k = 0
    model.generation_config.top_p = 1.0
    model.generation_config.repetition_penalty = REPETITION_PENALTY

    predict_model_dataset_otsu(tokenizer, model, counterfactual_dataset_dict, "llama8_counterfactual_conformal_sampling")
    return


if __name__ == "__main__":
    predict_llama_8b_counterfactual_conformal_sampling()
