import transformers

from common.params import *
from predict.otsu.predict_model_dataset_otsu import predict_model_dataset_otsu
from preprocess.preprocess_natural_instructions import preprocess_natural_instructions


def predict_qwen_14b_web_questions_e_sampling():
    tokenizer = transformers.AutoTokenizer.from_pretrained(QWEN_14B)
    model = transformers.AutoModelForCausalLM.from_pretrained(QWEN_14B, dtype="auto", device_map="auto")

    model.generation_config.do_sample = True
    model.generation_config.max_new_tokens = MAX_NEW_TOKENS_WEB_QUESTIONS
    model.generation_config.min_new_tokens = MIN_NEW_TOKENS_WEB_QUESTIONS
    model.generation_config.temperature = TEMPERATURE
    model.generation_config.top_k = 0
    model.generation_config.top_p = 1.0
    model.generation_config.epsilon_cutoff = EPSILON_CUTOFF
    model.generation_config.repetition_penalty = REPETITION_PENALTY

    web_questions_dataset_dict = preprocess_natural_instructions(
        tokenizer,
        "task1412_web_questions_question_answering"
    )
    predict_model_dataset_otsu(tokenizer, model, web_questions_dataset_dict, "qwen14_web_questions_e_sampling")
    return


if __name__ == "__main__":
    predict_qwen_14b_web_questions_e_sampling()
