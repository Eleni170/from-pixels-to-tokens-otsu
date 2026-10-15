import transformers

from common.params import *
from predict.otsu.predict_model_dataset_otsu import predict_model_dataset_otsu
from preprocess.preprocess_natural_instructions import preprocess_natural_instructions
from otsu_thresholding.generation.otsu_thresholding_model_for_causal_lm import OtsuThresholdingModelForCausalLM


def predict_qwen_8b_counterfactual_otsu_top_k():
    tokenizer = transformers.AutoTokenizer.from_pretrained(QWEN_8B)
    model = OtsuThresholdingModelForCausalLM.from_pretrained(QWEN_8B, dtype="auto", device_map="auto")

    model.generation_config.do_sample = True
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS_COUNTERFACTUAL
    model.generation_config.min_new_tokens = MIN_NEW_TOKENS_COUNTERFACTUAL
    model.generation_config.temperature = TEMPERATURE
    model.generation_config.top_k = TOP_K
    model.generation_config.top_p = 1.0
    model.generation_config.repetition_penalty = REPETITION_PENALTY

    counterfactual_dataset_dict = preprocess_natural_instructions(
        tokenizer,
        "task269_csrg_counterfactual_story_generation"
    )
    predict_model_dataset_otsu(tokenizer, model, counterfactual_dataset_dict, "qwen8_counterfactual_otsu_top_k")
    return


if __name__ == "__main__":
    predict_qwen_8b_counterfactual_otsu_top_k()
