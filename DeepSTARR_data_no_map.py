import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from FASTA_utils import parse_fasta, parse_header, plot_pca_2d, shuffle_dict,plot_pca_2d_labels
from tqdm import tqdm
import pickle
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


tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species",cache_dir="/media/zebrafish/Data2/Arman/Seq2Im_model_cache/")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species",cache_dir="/media/zebrafish/Data2/Arman/Seq2Im_model_cache/")

# Time to shuffle the sequences

Train_sequences, labels = shuffle_dict(Train_sequences, labels)


# Time to tokenize the sequences

tokenized_sequences = []
mean_embeddings = []
max_embeddings = []
for dna in tqdm(Train_sequences['Sequence']):
    if len(mean_embeddings) > 100:
        break
    tokenized_sequences.append(tokenizer(dna, return_tensors="pt")['input_ids'])
    hidden_states = model(tokenized_sequences[-1])[0]
    mean_embeddings.append(torch.mean(hidden_states[0], dim=0).detach().numpy())
    max_embeddings.append(torch.max(hidden_states[0], dim=0).values.detach().numpy())

print("Number of tokenized sequences: ", len(tokenized_sequences))
print("shape of the mean embedding: ", mean_embeddings[0].shape)
print("shape of the max embedding: ", max_embeddings[0].shape)



file_path = "/media/zebrafish/Data2/Arman/Seq2Im"
plot_pca_2d_labels(mean_embeddings,labels[:101],file_path,"mean_embeddings.png")
plot_pca_2d_labels(max_embeddings,labels[:101],file_path,"max_embeddings.png")
data_dict = {'Sequence':Train_sequences['Sequence'],'labels':labels,'mean_embeddings':mean_embeddings,'max_embeddings':max_embeddings}

file_name = "DeepSTARR_data_no_map.pkl"
with open(file_path+"/"+file_name, 'wb') as fp:
    pickle.dump(data_dict,fp)
print("The data has been saved to: ", os.path.join(file_path,file_name))