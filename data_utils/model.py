from transformers import CLIPModel, AutoModel, CLIPProcessor
from preprocessing import fcgr, protein_to_dna
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from CLIP.clip import clip
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
#_Tokenizer().decode(tokens=clip.tokenize(list(labels[2])[0]).numpy().tolist()[0])
lang = 'en'
config = {'tokenizer_file': 'tokenizer.json'
          }

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_all_sentences(ds):
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

    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

class image_title_dataset(Dataset):
    def __init__(self, processor, images, labels, max_len):
        # Initialize image paths and corresponding texts
        self.images = images
        self.max_len = max_len
        # Tokenize text using CLIP's tokenizer
        self.labels = labels
        self.processor = processor
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        labels_short = self.labels[idx:idx + 1].tolist()[:][:self.max_len]

        tokens = clip.tokenize(self.labels[idx:idx + 1].tolist())[:1, :self.max_len]
        if tokens.size()[1] < self.max_len:
            tokens = torch.cat([tokens,
                       torch.zeros(size=(1, self.max_len - tokens.size()[1]))], dim=1).type(
                torch.int)
        #attmask = torch.zeros()
        #pixel_values = torch.transforms

        inputs = self.processor(text=labels_short, images=self.images[idx:idx+1], return_tensors="pt", padding=True)
        inputs['input_ids'] = tokens
        if inputs['attention_mask'].size()[1] > self.max_len:
            inputs['attention_mask'] = inputs['attention_mask'][:1, :self.max_len]
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.zeros(size=(1, len(inputs['input_ids'][0]) - len(inputs['attention_mask'][0])))], dim=1).type(torch.int)

        #inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.Tensor([[0] * (self.max_len - len(inputs['input_ids'][0]))])], dim=1)
        #inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.Tensor([[0] * (self.max_len - len(inputs['attention_mask'][0]))])], dim=1)
        return inputs


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.requires_grad:
            p.grad.data = p.grad.data.float()

def train(model, processor, images, labels):
    tokenizer = get_or_build_tokenizer(labels)
    criterion = nn.CrossEntropyLoss()
    #for img in images:

    #labeltokens = clip.tokenize(labels.tolist(), 128)

    #labeltokens = np.array([np.array(tokenizer.encode(label).ids) for label in labels])

    #inputs = processor(text=labeltokens, images=images, return_tensors='pt', padding=True)
    #inputs = processor(text=torch.Tensor(labeltokens[:1]), images=torch.permute(images[:1], dims=(0,2,3,1)) )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=1e-7, weight_decay=0.0001)
    dataset = image_title_dataset(processor, images, labels, 77)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for i, inputs in enumerate(loader):
    #inputs = processor(text=labels[:].tolist(), images=images[:], return_tensors="pt", padding=True)

        optimizer.zero_grad()
        outputs = model(input_ids=torch.Tensor(inputs['input_ids']).type(torch.int),
            attention_mask=torch.Tensor(inputs['attention_mask']).squeeze(1), pixel_values=torch.Tensor(inputs['pixel_values']).squeeze(1))

        logits_i = outputs.logits_per_image
        logits_t = outputs.logits_per_text
        #probs = logits.softmax(dim=1)

        labels = torch.arange(0, logits_i.shape[0])
        loss_i = criterion(logits_i, labels)
        loss_t = criterion(logits_t, labels)

        loss = (loss_i + loss_t)/2
        loss.backward()

        if device == 'cpu':
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        print(loss.item())


    # outputs = model(torch.Tensor(labeltokens[0]), torch.permute(images[0], dims=(1,2,0)))
    #inputs = processor(text=labels, images=imgs, return_tensors='pt', padding=True)
    #outputs = model(**inputs)
    logits = outputs.logits_per_image
    probs = logits.softmax(dim=1)

        #print(probs)

# Example usage
#protein_seq = "MVK"
#print("DNA Sequence:", dna_seq)






print('finished')

if __name__ == '__main__':
    nsamples = 100
    labelcol = "Protein names"
    inputcol = "Sequence"
    labels = pd.read_csv(r"..//data//labels.csv")[labelcol][:nsamples]
    inputs = pd.read_csv(r"..//data//sequences.csv")[inputcol][:nsamples]
    imgs = np.array([fcgr(seq, k=7) for seq in inputs])
    imgs = torch.Tensor(imgs).unsqueeze(1).repeat(1, 3, 1, 1)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    #model = CLIPModel._get_args

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train(model, processor, imgs, labels)