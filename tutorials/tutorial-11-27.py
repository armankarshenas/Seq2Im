import torch
from transformers import AdamW,AutoTokenizer,AutoModelForSequenceClassification
from datasets import load_dataset
#input_txt = ["My name is Arman","You are a nice person","I hate this"]
#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#model=AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
#input_tokens = tokenizer(input_txt,padding=True,truncation=True,return_tensors="pt")
#print(input_tokens)
#input_tokens["labels"] = torch.tensor([1,1,0])
#print(input_tokens)
#model.config.pad_token = model.config.eos_token_id
#print(model.config.pad_token)
#optimizer = torch.optim.AdamW(model.parameters())
#loss = model(**input_tokens).loss
#loss.backward()
#optimizer.step()

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets

raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]


raw_train_dataset.features

from transformers import AutoTokenizer,DataCollatorWithPadding
from torch.utils.data import DataLoader

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets.rename_column("label","labels")
tokenized_datasets.remove_columns(["idx","sentence1","sentence2"])
tokenized_datasets.with_format("torch")
print(tokenized_datasets)
data_collator = DataCollatorWithPadding(tokenizer)
train_loader = DataLoader(tokenized_datasets["train"],batch_size=16,shuffle=True,collate_fn=data_collator)
print(train_loader)
