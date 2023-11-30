from transformers import CLIPModel, AutoModel, CLIPProcessor
from preprocessing import fcgr, protein_to_dna
import torch

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
label_column = 'Protein names'
input_column = 'Sequence'

df = pd.read_table(r'..//data//uniprotkb_taxonomy_id_9606_AND_model_or_2023_11_30.tsv')

# Example usage
#protein_seq = "MVK"
sequences = df[input_column]
labels = df[label_column]

labels = sequences[~sequences.str.contains('U|O')]
sequences = sequences[~sequences.str.contains('U|O')]

dna_sequences = sequences.apply(protein_to_dna)
#print("DNA Sequence:", dna_seq)

imgs = np.array([fcgr(dna_seq, k=3) for dna_seq in dna_sequences])
imgs = torch.Tensor(imgs).unsqueeze(1).repeat(1, 3, 1, 1)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(text=['dog', 'cat'], images=img, return_tensors='pt', padding=True)
outputs = model(**inputs)
logits = outputs.logits_per_image
probs = logits.softmax(dim=1)

print(probs)

print('finished')

