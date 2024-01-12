import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model
import os
import torch
from FASTA_utils import parse_fasta, parse_header, shuffle_dict,plot_pca_2d_labels
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


Train_sequences, labels = shuffle_dict(Train_sequences, labels)

sequences = Train_sequences['Sequence'][0:1000]
label_subset = labels[0:1000]
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name="250M_multi_species_v2",
    embeddings_layers_to_save=(20,),
    max_positions=249,
)
forward_fn = hk.transform(forward_fn)

# Get data and tokenize it
tokens_ids = [b[1] for b in tqdm(tokenizer.batch_tokenize(sequences))]
tokens_str = [b[0] for b in tqdm(tokenizer.batch_tokenize(sequences))]
tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

# Initialize random key
random_key = jax.random.PRNGKey(0)

# Infer
outs = forward_fn.apply(parameters, random_key, tokens)

# Get embeddings at layer 20
print(outs["embeddings_20"].shape)