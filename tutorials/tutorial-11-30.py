from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,DataCollatorWithPadding
from datasets import load_dataset,metric
import torch
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained("test-trainer/checkpoint-1000")
print(model.parameters())
raw_data = load_dataset("glue","mrpc")
tokenizer_using = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_fn(example):
    return tokenizer_using(example["sentence1"],example["sentence2"],truncation=True)
tokenized_data = raw_data.map(tokenize_fn,batched=True)
training_data = tokenized_data["train"]
training_data = training_data.remove_columns(["idx","sentence1","sentence2"])
training_data = training_data.rename_column("label","labels")
training_data = training_data.with_format("torch")
testing_data = tokenized_data["test"]
testing_data = testing_data.remove_columns(["idx","sentence1","sentence2"])
testing_data = testing_data.rename_column("label","labels")
testing_data = testing_data.with_format("torch")
print(testing_data)
datacollator = DataCollatorWithPadding(tokenizer = tokenizer_using)
trainer = Trainer(model=model,data_collator=datacollator)

pred = trainer.predict(testing_data)
print(pred)
pred_labels = np.argmax(pred.predictions,axis=-1)
import evaluate

metric = evaluate.load("glue","mrpc")
print(metric.compute(predictions=pred_labels,references=pred.label_ids))

# Now implementing a full training using torch loop
from transformers import AdamW
from torch.utils.data import DataLoader

training_loader = DataLoader(tokenized_data["train"],batch_size=16,shuffle=True,collate_fn=datacollator)
evaluation_loader = DataLoader(tokenized_data["validation"],batch_size=16,collate_fn=datacollator)

for batch in training_loader:
    break
{k:v.shape() for k,v in batch.items()}

