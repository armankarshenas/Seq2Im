import torch
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,Trainer,TrainingArguments
from torch.utils.data import DataLoader
data_raw = load_dataset("glue","mrpc")
print(data_raw)
tokenizer_using = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_fn(example):
    return tokenizer_using(example["sentence1"],example["sentence2"],truncation=True)
tokenized_dataset = data_raw.map(tokenize_fn,batched=True)
print(tokenized_dataset["train"][0])
tokenized_dataset = tokenized_dataset.rename_column("label","labels")
tokenized_dataset = tokenized_dataset.remove_columns(["idx","sentence1","sentence2"])
print(tokenized_dataset)
data_collator = DataCollatorWithPadding(tokenizer_using)
Training_loader = DataLoader(tokenized_dataset["train"],batch_size=1,shuffle=True)
print(Training_loader)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2)
Train_arg = TrainingArguments("test-trainer")

trainer = Trainer(model,Train_arg,data_collator=data_collator,train_dataset=tokenized_dataset["train"],eval_dataset=tokenized_dataset["validation"],tokenizer=tokenizer_using)
trainer.train()