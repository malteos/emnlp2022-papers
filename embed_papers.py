#!/usr/bin/env python

import argparse
import os
import json

import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
import umap


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default=None, type=str, help='Path to Excel sheet downloaded from EMNLP')
    parser.add_argument('--output_path', default=None, type=str, help='Path to save output JSON file')
    parser.add_argument('--model_name_or_path', default='malteos/scincl', type=str, help='Model used for generating the paper embeddings')
    parser.add_argument('--limit', default=0, type=int, help='Limit input samples')
    parser.add_argument('--batch_size', default=8, type=int, help='Limit input samples')
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_excel(args.input_path)  
    labels = list(sorted(df['Track'].unique()))

    if args.limit > 0:
        df = df.sample(n=args.limit).reindex()

    print(df)
    print(df.columns)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    title_abs = []
    for idx, row in df.iterrows():
        title_abs.append(row['Title'] + tokenizer.sep_token + (row['Abstract'] if isinstance(row['Abstract'], str) else ''))

    print(len(title_abs))

    # load model and tokenizer
    model = AutoModel.from_pretrained(args.model_name_or_path)

    # preprocess the input
    model_inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)

    class PaperDataset(Dataset):
        def __init__(self, tokenizer_out):
            self.tokenizer_out = tokenizer_out
            
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.tokenizer_out.items()}
        
        def __len__(self):
            return len(self.tokenizer_out['input_ids'])

    ds = PaperDataset(model_inputs)    
    dl = DataLoader(ds, batch_size=args.batch_size)

    model = DataParallel(model)
    model.eval()
    
    # inference
    embeddings = []
    with torch.no_grad():
            for step, inputs in enumerate(tqdm(dl, desc='Inference')):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model_out = model(**inputs)

                # take the first token ([CLS] token) in the batch as the embedding
                step_embeds = model_out.last_hidden_state[:, 0, :].cpu().numpy()

                embeddings += step_embeds.tolist()
    
    embeddings = np.array(embeddings)

    print('umap')

    # maybe make smaller to focus on local structure, Sensible values are in the range 0.001 to 0.5, 
    # metric='cosine' # correlation
    embeddings_2d = umap.UMAP(
        n_neighbors=10, #20,
        min_dist=0.05, #0.5,  
        random_state=1, # good = 1
        ).fit_transform(embeddings)
            
    # output
    papers = []
    for idx, (_, row) in enumerate(df.iterrows()):
        # {"loc":[41.575330,13.102411], "title":"aquamarine"},
        papers.append({
            'loc': embeddings_2d[idx].tolist(),
            'id': row['Submission ID'],
            'title': row['Title'],
            'authors': row['Authors'],
            'abstract': (row['Abstract'] if isinstance(row['Abstract'], str) else ''),
            'track': row['Track'],
            'label': labels.index(row['Track'])
        })

    print('labels: ', labels, len(labels))

    # save
    with open(args.output_path, 'w') as f:
        json.dump(papers, f)

    print('done')
