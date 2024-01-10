import os
from transformers import AutoTokenizer, AutoModel,TrainingArguments, Trainer
import torch
from FASTA_utils import parse_fasta, parse_header
from tqdm import tqdm
from datasets import Dataset
import numpy as np


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


tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")

# Time to tokenize the sequences
data_train = Dataset.from_dict({'Sequence':Train_sequences['Sequence'],'labels':labels})


tokenized_sequences = []
mean_embeddings = []
max_embeddings = []
for dna in tqdm(Train_sequences['Sequence']):
    tokenized_sequences.append(tokenizer(dna, return_tensors="pt")['input_ids'])
    hidden_states = model(tokenized_sequences[-1])[0]
    mean_embeddings.append(torch.mean(hidden_states[0], dim=0))
    max_embeddings.append(torch.max(hidden_states[0], dim=0))
