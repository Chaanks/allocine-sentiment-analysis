#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "0.1.0"
__maintainer__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]

import utils

from pathlib import Path
import json
import spacy
import pandas as pd
from tqdm import tqdm
from loguru import logger


def create_dataset(name: str , filename: Path, output_dir: Path, nlp=None):
    """ extract json dataset to a specified path in .txt"""
    
    # create output dir
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Loading raw json file
    with open(filename, 'r', encoding='utf8') as file:   
        raw_dataset = json.load(file)
    
    # write files
    for _, review in enumerate(tqdm(raw_dataset)):

        # filter and standardize text
        filtered = utils.filter_comment(review['commentaire'], nlp)
        if len(filtered) == 0:
            logger.warning(f"remove {review['review_id']}")
            continue

        label_dir = dataset_dir / str(review['note'])
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = label_dir / f"{review['review_id'].replace('review_', '')}.txt"
        filename.touch()
        with open(filename, 'w') as f:
            f.write(filtered)


def create_scoring(name: str , filename: Path, output_dir: Path, nlp=None):
    # create output dir
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Loading raw json file
    with open(filename, 'r', encoding='utf8') as file:   
        raw_dataset = json.load(file)

    reviews = {'review_id': [], 'commentaire': []}
    # write files
    for _, review in enumerate(tqdm(raw_dataset)):

        # filter and standardize text
        filtered = utils.filter_comment(review['commentaire'], nlp)
        if len(filtered) == 0:
            logger.warning(f"empty review {review['review_id']}")
            reviews['review_id'].append(review['review_id'])
            reviews['commentaire'].append(review['commentaire'])
            continue

        reviews['review_id'].append(review['review_id'])
        reviews['commentaire'].append(filtered)
    
    df = pd.DataFrame.from_dict(reviews)
    df.to_csv(dataset_dir / 'trials.csv')

if __name__ == '__main__':

    # Loading SpaCy model from tokenization utilities
    spacy.prefer_gpu()
    nlp = spacy.load('fr_core_news_sm')

    # Loading stop words
    stop_words_path = Path(
        '/home/jarod/git/allocine-sentiment-analysis/data/json/stopwords.json'
    )
    with open(stop_words_path, 'r', encoding='utf8') as file:
        stop_words = json.load(file)
    nlp.Defaults.stop_words = set(stop_words)

    output_dir = Path("../data/allocine_filtered")
    train_json = Path("../data/json/train.json")
    eval_json = Path("../data/json/dev.json")
    test_json = Path("../data/json/test.json")

    # create_dataset('eval', eval_json, output_dir, nlp=nlp)
    # create_dataset('train', train_json, output_dir, nlp=nlp)

    create_scoring('test', test_json, output_dir, nlp=nlp)

