# 加载预训练数据
from datasets import load_dataset

ds = load_dataset('json', data_files='data/mobvoi_seq_monkey_general_open_corpus.clean.jsonl')


column_names = list(ds["train"].features)
print(column_names)