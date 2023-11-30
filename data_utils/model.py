from transformers import CLIPModel, AutoModel, CLIPProcessor
from preprocessing import fcgr, protein_to_dna
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

lang = 'en'
config = {'tokenizer_file': 'tokenizer.json'
          }

def get_all_sentences(ds, lang):
    ''' dataset consists of pairs of sentences. Get only part that is in language <lang>'''
    for item in ds:
        yield item

def get_or_build_tokenizer(ds):
    '''

    :param config: dict with configs
    :param ds: dataset
    :param lang: language
    :return: tokenizer for given language dataset and configurations
    '''

    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def train(model, processor, images, labels):
    tokenizer = get_or_build_tokenizer(labels)
    loss = nn.CrossEntropyLoss()
    for img in images:
        inputs = processor(text=labels, images=img, return_tensors='pt', padding=True)
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)
        inputs = processor(text=labels, images=imgs, return_tensors='pt', padding=True)
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)

        #print(probs)

# Example usage
#protein_seq = "MVK"
#print("DNA Sequence:", dna_seq)






print('finished')

if __name__ == '__main__':
    labelcol = "Protein names"
    inputcol = "Sequence"
    labels = pd.read_csv(r"..//data//labels.csv")[labelcol]
    inputs = pd.read_csv(r"..//data//sequences.csv")[inputcol]
    imgs = np.array([fcgr(seq, k=3) for seq in inputs])
    imgs = torch.Tensor(imgs).unsqueeze(1).repeat(1, 3, 1, 1)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    train(model, processor, imgs, labels)