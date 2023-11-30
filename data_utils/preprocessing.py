import numpy as np
from matplotlib import pyplot as plt

def protein_to_dna(protein_sequence):
    # Genetic code dictionary
    genetic_code = {
        'F': ['TTT', 'TTC'], 'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
        'I': ['ATT', 'ATC', 'ATA'], 'M': ['ATG'], 'V': ['GTT', 'GTC', 'GTA', 'GTG'],
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], 'P': ['CCT', 'CCC', 'CCA', 'CCG'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'], 'A': ['GCT', 'GCC', 'GCA', 'GCG'],
        'Y': ['TAT', 'TAC'], 'H': ['CAT', 'CAC'], 'Q': ['CAA', 'CAG'],
        'N': ['AAT', 'AAC'], 'K': ['AAA', 'AAG'], 'D': ['GAT', 'GAC'],
        'E': ['GAA', 'GAG'], 'C': ['TGT', 'TGC'], 'W': ['TGG'],
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'],
        '*': ['TAA', 'TAG', 'TGA'], 'X': ['TAA', 'TAG', 'TGA']  # Stop codon
    }

    dna_sequence = ''
    for amino_acid in protein_sequence:
        if amino_acid in genetic_code:
            # Select the first codon for simplicity
            dna_sequence += genetic_code[amino_acid][0]
        else:
            raise ValueError(f"Invalid amino acid '{amino_acid}' found in the input sequence.")

    return dna_sequence

def fcgr(seq: str, k=8):
    letter_to_num = {'A': 0,
                    'C': 1,
                    'G': 2,
                    'T': 3}

    letter_to_x = {'A': 0,
                    'C': 1,
                    'G': 0,
                    'T': 1}

    letter_to_y = {'A': 0,
                    'C': 0,
                    'G': 1,
                    'T': 1}

    IMGSIZE = 2**k
    img = np.zeros((IMGSIZE, IMGSIZE))

    substrs = [seq[i:i+k] for i in range(len(seq)-k+1)]

    for substr in substrs:
        x = 0
        y = 0
        for i, s in enumerate(substr):
            x = x + letter_to_x[s] * IMGSIZE/(2 ** (i+1))
            y = y + letter_to_y[s] * IMGSIZE / (2 ** (i+1))
        img[int(y), int(x)] += 1

    return img


def preprocess_inputs(path, path_save):
    label_column = 'Protein names'
    input_column = 'Sequence'
    df = pd.read_table(path)
    sequences = df[input_column]
    labels = df[label_column]

    labels = sequences[~sequences.str.contains('U|O')]
    sequences = sequences[~sequences.str.contains('U|O')]
    dna_sequences = sequences.apply(protein_to_dna)

    labels.save(path_save + 'labels.csv')
    dna_sequences.save(path_save + 'sequences.csv')
    print('saved processed sequences and labels (annotations')

if __name__ == '__main__':
    preprocess_inputs(r'data//uniprotkb_taxonomy_id_9606_AND_model_or_2023_11_30.tsv',
                      r'data//')
    #img = fcgr('ACG', k=3)
    #print(img,)
    #plt.imshow(img)
    #plt.show()