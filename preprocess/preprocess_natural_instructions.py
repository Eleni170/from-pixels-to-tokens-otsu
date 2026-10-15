import datasets

from common.params import *


def _preprocess_natural_instructions(batch, tokenizer):
    contexts = []
    texts = []
    targets = []
    for definition, inputs, target in zip(batch["definition"], batch["inputs"], batch["targets"]):
        contexts.append(inputs)
        message = [{
            "role": "user",
            "content": f"### Instructions: {definition}\n\n### Input: {inputs}\n\n### Output: {tokenizer.eos_token}"
        }]
        texts.append(
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        )
        targets.append(target + tokenizer.eos_token)

    contexts_tokenized = tokenizer(contexts, padding=True, padding_side="left")
    texts_tokenized = tokenizer(texts, padding=True, padding_side="left")
    targets_tokenized = tokenizer(text_target=targets, padding=True, padding_side="left")

    return {
        "contexts": contexts_tokenized["input_ids"],
        "input_ids": texts_tokenized["input_ids"],
        "attention_mask": texts_tokenized["attention_mask"],
        "labels": targets_tokenized["input_ids"]
    }


def preprocess_natural_instructions(tokenizer, task_name):
    natural_instructions_dataset_dict = datasets.DatasetDict()

    natural_instructions_dataset = datasets.load_dataset("Muennighoff/natural-instructions")["train"]
    natural_instructions_dataset = natural_instructions_dataset.filter(lambda x: x["task_name"] == task_name)
    natural_instructions_dataset = natural_instructions_dataset.select(range(3333))

    natural_instructions_test_dataset = natural_instructions_dataset.select(range(3000))
    natural_instructions_val_dataset = natural_instructions_dataset.select(
        range(3000, len(natural_instructions_dataset))
    )

    natural_instructions_dataset_map = natural_instructions_test_dataset.map(
        _preprocess_natural_instructions,
        batched=True,
        batch_size=BATCH_SIZES["test_otsu"],
        remove_columns=natural_instructions_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

    natural_instructions_dataset_dict["test"] = natural_instructions_dataset_map
    natural_instructions_dataset_dict["test"].set_format("torch")

    natural_instructions_dataset_map = natural_instructions_val_dataset.map(
        _preprocess_natural_instructions,
        batched=True,
        batch_size=BATCH_SIZES["val_otsu"],
        remove_columns=natural_instructions_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

    natural_instructions_dataset_dict["val"] = natural_instructions_dataset_map
    natural_instructions_dataset_dict["val"].set_format("torch")

    return natural_instructions_dataset_dict
