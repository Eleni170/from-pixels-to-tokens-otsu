import transformers

from common.params import *
from predict.otsu.predict_model_dataset_otsu import predict_model_dataset_otsu
from preprocess.preprocess_natural_instructions import preprocess_natural_instructions


def predict_llama_8b_extreme_abstract_top_p_sampling():
    tokenizer = transformers.AutoTokenizer.from_pretrained(LLAMA_8B)
    tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
    model = transformers.AutoModelForCausalLM.from_pretrained(LLAMA_8B, dtype="auto", device_map="auto")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    model.generation_config.do_sample = True
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS_EXTREME_ABSTRACT
    model.generation_config.min_new_tokens = MIN_NEW_TOKENS_EXTREME_ABSTRACT
    model.generation_config.temperature = TEMPERATURE
    model.generation_config.top_k = TOP_K
    model.generation_config.top_p = TOP_P
    model.generation_config.repetition_penalty = REPETITION_PENALTY

    extreme_abstract_dataset_dict = preprocess_natural_instructions(
        tokenizer,
        "task668_extreme_abstract_summarization"
    )
    predict_model_dataset_otsu(tokenizer, model, extreme_abstract_dataset_dict, "llama8_extreme_abstract_top_p_sampling")
    return


if __name__ == "__main__":
    predict_llama_8b_extreme_abstract_top_p_sampling()
