import os
from transformers import AutoTokenizer, AutoModelForMaskedLM,DataCollatorForLanguageModeling
import torch
from FASTA_utils import parse_fasta, parse_header
from tqdm import tqdm
from datasets import Dataset
import numpy as np

torch.set_default_device("cuda")
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


tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species",cache_dir="/media/zebrafish/Data2/Arman/Seq2Im_model_cache/")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species",cache_dir="/media/zebrafish/Data2/Arman/Seq2Im_model_cache/")

# Time to tokenize the sequences
data_train = Dataset.from_dict({'Sequence':Train_sequences['Sequence'],'labels':labels})
print(data_train[0])

def tokenize_function(examples):
    return tokenizer(examples['Sequence'], return_tensors="pt", truncation=True, padding="max_length", max_length=249)
data_tokenizations = data_train.map(tokenize_function, batched=True)
print(data_tokenizations)




