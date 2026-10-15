import transformers

from common.params import *
from predict.otsu.predict_model_dataset_otsu import predict_model_dataset_otsu
from preprocess.preprocess_natural_instructions import preprocess_natural_instructions


def predict_mistral_24b_counterfactual_typical_sampling():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MISTRAL_24B, fix_mistral_regex=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model = transformers.AutoModelForCausalLM.from_pretrained(MISTRAL_24B, dtype="auto", device_map="auto")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    model.generation_config.do_sample = True
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS_COUNTERFACTUAL
    model.generation_config.min_new_tokens = MIN_NEW_TOKENS_COUNTERFACTUAL
    model.generation_config.temperature = TEMPERATURE
    model.generation_config.top_k = 0
    model.generation_config.top_p = 1.0
    model.generation_config.repetition_penalty = REPETITION_PENALTY

    counterfactual_dataset_dict = preprocess_natural_instructions(
        tokenizer,
        "task269_csrg_counterfactual_story_generation"
    )
    predict_model_dataset_otsu(tokenizer, model, counterfactual_dataset_dict, "mistral24_counterfactual_t_sampling")
    return


if __name__ == "__main__":
    predict_mistral_24b_counterfactual_typical_sampling()
