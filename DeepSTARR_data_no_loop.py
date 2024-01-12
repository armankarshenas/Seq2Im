import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from FASTA_utils import parse_fasta, parse_header, plot_pca_2d, shuffle_dict,plot_pca_2d_labels
from tqdm import tqdm
import pickle
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species",cache_dir="/media/zebrafish/Data2/Arman/Seq2Im/")
model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species",cache_dir="/media/zebrafish/Data2/Arman/Seq2Im/")

Train_sequences, labels = shuffle_dict(Train_sequences, labels)

Seq_subset = Train_sequences['Sequence'][:100]
labels_subset = labels[:100]


# Time to tokenize the sequences

tokenized_sequences_ids = tokenizer.batch_encode_plus(Seq_subset, return_tensors="pt",truncation=True,padding="max_length",max_length=249)['input_ids']
attention_mask = (tokenized_sequences_ids != tokenizer.pad_token_id)
print("Tokenization is done!")

torch_output = model(tokenized_sequences_ids,attention_mask=attention_mask,encoder_attention_mask = attention_mask,output_hidden_states=True)
print("Modeling is done!")

embeddings = torch_output['hidden_states'][-1].detach().numpy()
print("Embeddings are extracted!")
print("shape of the embedding: ", embeddings.shape)
print("an example of the embedding: ", embeddings[0])

attention_mask = torch.unsqueeze(attention_mask, dim = -1)


