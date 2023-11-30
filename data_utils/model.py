from transformers import CLIPModel, AutoModel, CLIPProcessor
from preprocessing import fcgr
import torch

import pandas as pd


df = pd.read_table(r'..//data//uniprotkb_taxonomy_id_9606_AND_model_or_2023_11_30.tsv')

img = fcgr('ACG', k=3)
img = torch.Tensor(img).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

inputs = processor(text=['dog', 'cat'], images=img, return_tensors='pt', padding=True)
outputs = model(**inputs)
logits = outputs.logits_per_image
probs = logits.softmax(dim=1)

print(probs)

print('finished')

