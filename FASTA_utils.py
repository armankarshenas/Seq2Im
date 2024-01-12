import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import os
import random
def parse_fasta(fasta_file):
    fasta_dict = {'Sequence':[],'Header':[]}
    current_header = None
    current_sequence = ""

    with open(fasta_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # Header line
                if current_header is not None:
                    # Save the previous sequence
                    fasta_dict['Sequence'].append(current_sequence)
                    fasta_dict['Header'].append(current_header)
                current_header = line[1:]
                current_sequence = ""
            else:
                # Sequence line
                current_sequence += line

        # Save the last sequence
        if current_header is not None:
            fasta_dict['Sequence'].append(current_sequence)
            fasta_dict['Header'].append(current_header)

    return fasta_dict

def parse_header(dict):
    headers = dict['Header']
    labels = np.zeros(len(headers))
    for i,header in enumerate(headers):
        if "positive" in header:
            labels[i] = 1
        else:
            labels[i] = 0
    return labels

def plot_pca_2d(X,path,file_name):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)
    plt.title('2D PCA Plot')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
    plt.savefig(os.path.join(path,file_name))

def shuffle_dict(dict,labels):
    sequences = dict['Sequence']
    header = dict['Header']
    combined_list = list(zip(sequences,labels,header))
    random.shuffle(combined_list)
    shuffled_sequences,shuffled_labels,shuffled_header = zip(*combined_list)
    shuffled_dict = {'Sequence':list(shuffled_sequences),'Header':list(shuffled_header)}
    return shuffled_dict,list(shuffled_labels)

def plot_pca_2d_labels(X, Y,path,file_name):
    # Apply PCA to reduce dimensionality to 2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap='viridis', edgecolors='k', s=60)

    # Add labels and title
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA of Observations Colored by Labels')

    # Add legend
    legend_labels = ['Class 0', 'Class 1']
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, loc='upper right')

    # Display the plot
    plt.show()
    plt.savefig(path +"/"+file_name)
