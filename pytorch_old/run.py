#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run.py: This file is used as a launcher to train our model.
"""

__author__ = "Jonathan Heno, Duret Jarod"
__license__ = "MIT"

from pathlib import Path

from dataset import AlloCine


if __name__ == "__main__":
    ds_path = Path('/home/jarod/git/allocine-sentiment-analysis/data/csv/train_1000.csv')

    ds_toy = AlloCine(ds_path)
    print(ds_toy)

    test = ds_toy.__getitem__(0)
    print(test.shape)