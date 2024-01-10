import numpy as np


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
    for header in headers:
        if header.contains("positive"):
            labels[headers.index(header)] = 1
        else:
            labels[headers.index(header)] = 0
    return labels