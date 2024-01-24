from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer
import os
from FASTA_utils import parse_fasta, parse_header, shuffle_dict
import torch
from sklearn.metrics import matthews_corrcoef,f1_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from peft import LoraConfig, TaskType
from datasets import Dataset, load_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

Path = "/media/zebrafish/Data2/Arman/DeepSTARR-data/"
file_name = "Sequences_Train.fa"
Train_sequences = parse_fasta(os.path.join(Path,file_name))
labels = parse_header(Train_sequences)
print("Example sequence: ", (Train_sequences['Sequence'][0]))
print("Example header: ", (Train_sequences['Header'][0]))
print("Example label: ", labels[0])
print("Number of sequences: ", len(Train_sequences['Sequence']))
print("Number of headers: ", len(Train_sequences['Header']))
print("Sequence length: ", len(Train_sequences['Sequence'][0]))


num_labels =2
model = AutoModelForSequenceClassification.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",num_labels=num_labels,cache_dir="/media/zebrafish/Data2/Arman/Seq2Im_model_cache/")
model.to(device)

from peft import get_peft_model
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS,inference_mode=False, r=1, lora_alpha= 32, lora_dropout=  0.1, target_modules=["query","values"])
lora_classifier = get_peft_model(model, peft_config)
lora_classifier.print_trainable_parameters()
lora_classifier.to(device)
print(lora_classifier)
example_data = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks","promoter_all",split="train",streaming=False)

print("Example data structure: ", example_data)

train_sequences = example_data['sequence']
train_labels = example_data['label']

train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(train_sequences, train_labels, test_size=0.05, random_state=42)

idx_sequence = -1
sequence, label = train_sequences[idx_sequence], train_labels[idx_sequence]
print(f"The DNA sequence is {sequence}.")
print(f"Its associated label is label {label}.")

tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",cache_dir="/media/zebrafish/Data2/Arman/Seq2Im_model_cache/")

ds_train_promoter = Dataset.from_dict({"data": train_sequences,'labels':train_labels})
ds_validation_promoter = Dataset.from_dict({"data": validation_sequences,'labels':validation_labels})

def tokenize_function(examples):
    return tokenizer(examples['data'], return_tensors="pt", truncation=True, padding="max_length", max_length=249)

tokenized_train_data = ds_train_promoter.map(tokenize_function, batched=True,remove_columns=['data'])
tokenized_validation_data = ds_validation_promoter.map(tokenize_function, batched=True,remove_columns=['data'])

batch_size = 8
model_name='nucleotide-transformer'
args_promoter = TrainingArguments(
    f"{model_name}-finetuned-lora-NucleotideTransformer",
    remove_unused_columns=False,
    evaluation_strategy="steps",
    save_strategy="steps",
    learning_rate=5e-4,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps= 1,
    per_device_eval_batch_size= 64,
    num_train_epochs= 2,
    logging_steps= 100,
    load_best_model_at_end=True,  # Keep the best model according to the evaluation
    metric_for_best_model="f1_score",
    label_names=["labels"],
    dataloader_drop_last=True,
    max_steps= 1000
)

def compute_metrics_f1_score(eval_pred):
    """Computes F1 score for binary classification"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r={'f1_score': f1_score(references, predictions)}
    return r

trainer = Trainer(model.to(device), args_promoter, train_dataset=tokenized_train_data, eval_dataset=tokenized_validation_data,compute_metrics=compute_metrics_f1_score)
train_result = trainer.train()











