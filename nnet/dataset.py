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
from torch.utils.data import Dataset


# I/O functions :
def load_n_col(file: Path):
    """ 
        Used to load files
    """
    df = pd.read_csv(file, header=None)
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

        self.features, self.targets = load_n_col(self.ds_path)


    def __repr__(self):
        return f"AlloCine Dataset of size {len(self)}"

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.features)


    def __getitem__(self, idx):
        """ Returns one random utt of selected speaker """
        return torch.FloatTensor([0.0, 0.0, 0.0, 0.0])