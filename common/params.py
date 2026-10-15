import torch
import pathlib

# Configuration parameters
ENTREZ_EMAIL = "e.partalidou@gmail.com"
HUGGING_FACE_TOKEN = ""

# Path parameters
PWD_PATH = str(pathlib.Path.home())
ROOT_PATH = PWD_PATH + "/PycharmProjects/from-pixels-to-tokens-otsu"
VENV_PATH = ROOT_PATH + "/.venv"

# Instruction models parameters
LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
MISTRAL_24B = "mistralai/Mistral-Small-24B-Instruct-2501"
QWEN_8B = "Qwen/Qwen3-8B"
QWEN_14B = "Qwen/Qwen3-14B"

# Paraphrasing parameters
PARAPHRASE_NUM_GENERATIONS = 3

# min/max new tokens per dataset task
MAX_NEW_TOKENS_AMAZON_REVIEW = 73
MAX_NEW_TOKENS_EXTREME_ABSTRACT = 103
MAX_NEW_TOKENS_COUNTERFACTUAL = 74
MAX_NEW_TOKENS_WEB_QUESTIONS = 44
MIN_NEW_TOKENS_AMAZON_REVIEW = 7
MIN_NEW_TOKENS_EXTREME_ABSTRACT = 2
MIN_NEW_TOKENS_COUNTERFACTUAL = 16
MIN_NEW_TOKENS_WEB_QUESTIONS = 1

# Text generation parameters
ACTIVATION_DROPOUT = 0.1
ATTENTION_PROBS_DROPOUT_PROB = 0.1
BATCH_SIZES = {
    "val_otsu": 16,
    "test_otsu": 16
}
CONDITIONING = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON_CUTOFF = 9e-4
HIDDEN_DROPOUT_PROB = 0.1
LEARNING_RATE = 5e-5
MAX_LENGTH_ARTICLE = 8192
NUM_BEAMS = 3
NUM_SUBSET_RECORDS = 10
NUM_TRAIN_EPOCHS = 3
OPTIM = "adamw_torch"
REPETITION_PENALTY = 1.0
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.80
