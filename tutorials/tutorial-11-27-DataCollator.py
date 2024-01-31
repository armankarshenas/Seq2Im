from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader

dataset_raw = load_dataset("glue","mrpc")
print(dataset_raw)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def arman(example):
    return tokenizer(example["sentence1"],example["sentence2"],truncation=True)
dataset_raw.map(arman,batched=True)
print(dataset_raw)
data_collator = DataCollatorWithPadding(tokenizer)
train_loader = DataLoader(dataset_raw["train"],batch_size=16,shuffle=True,collate_fn=data_collator)
print(train_loader)