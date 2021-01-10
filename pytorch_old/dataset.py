#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dataset.py: This file contains funtions to read movie files
and the class of our dataset.
"""

__author__ = "Heno Jonathan, Duret Jarod"
__license__ = "MIT" 


from pathlib import Path

import numpy as np
import pandas as pd
import torch
import spacy
from torch.utils.data import Dataset


SIZE = 2000
DIMENSION = 96

# I/O functions :
def load_n_col(file: Path):
    """ 
        Used to load files
    """
    df = pd.read_csv(file)
    columns = [np.array(df[col]) for col in df]
    return columns 


# Pre-process functions:



# Dataset classes :
class AlloCine(Dataset):
    """ Characterizes a dataset for Pytorch """
    def __init__(self, ds_path: Path):
        """ Initialization """
        self.ds_path = Path(ds_path)
        assert ds_path.is_file()

        self.nlp = spacy.load("fr_core_news_sm")
        self.idx, self.targets, self.features = load_n_col(self.ds_path)

    def __repr__(self):
        return f"AlloCine Dataset of size {len(self)}"

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.features)

    def __getitem__(self, idx):
        """ TODO """

        embed = torch.zeros(size=(SIZE, DIMENSION), dtype=torch.torch.float32)

        # convert to emdeddings using spacy
        tokens = self.nlp(''.join(self.features[idx]))
        for idx, token in enumerate(tokens):
            embed[idx] = torch.from_numpy(token.vector)
        
        return embed, self.targets[idx]