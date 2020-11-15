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
    ds_path = Path('/home/jarod/git/allocine-sentiment-analysis/data/toy.csv')

    ds_toy = AlloCine(ds_path)
    print(ds_toy)