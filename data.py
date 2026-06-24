from datasets import load_dataset
from transformers import AutoTokenizer
from itertools import chain

def tokenize_function(examples, tokenizer):
    output = tokenizer([item for item in examples["text"]])
    return output

block_size = 2048
num_proc=0

def group_texts(examples):
    concatenated_examples= {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {k: [t[i:i+block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()}
    result["labels"] = result["input_ids"].copy()
    return result

def main():

    
    tokenizer = AutoTokenizer.from_pretrained(
        "./qwen-1.5b",
        trust_remote_code=True
    )
    ds = load_dataset(
        "json",
        data_files="data/mobvoi_seq_monkey_general_open_corpus.clean.jsonl"
    )

    column_names = list(ds["train"].features)

    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=column_names,
        load_from_cache_file=True,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        desc="Running tokenizer on dataset",
    )

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
        batch_size = 40000,#这里是处理数据的batch，不是训练的batch
    )
    train_dataset = lm_datasets["train"]
    print(f"train_dataset: {train_dataset}")





if __name__ == "__main__":
    main()