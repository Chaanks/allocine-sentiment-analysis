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


import xml.etree.ElementTree
import numpy
import json
import pathlib
import re
import shutil
import math
import spacy

from tqdm import tqdm
from pathlib import Path, PurePath


REGEXES = [
    lambda x: re.sub(r'http\S+', ' ', x),
    lambda x: re.sub(r'[^\d|(a-z)|\U0001F600-\U0001F64F|!|.|#|@|è|é|à|ù|ô|ü|ë|ä|û|î|ê|â|ç\s]', ' ', x),
    lambda x: re.sub(r'<br />', ' ', x),
    lambda x: re.sub(r'[\(|\)]', ' ', x),
    #lambda x: re.sub(r'(.+?)\1{2,}', r'\1\1\1', x),
    lambda x: re.sub(r'\s+', ' ', x)
]


def filter_comment(comment: str, nlp=None, as_list: bool=False) -> str or list:
    """
    Filters out a comment with the list of regular expressions given in 
    `REGEXES`.

    Parameters
    ----------
    comment: str

    nlp: spacy.NLP (default: None)
    
    as_list: bool

    Returns
    -------
    str | list(str)
        The filtered comment as a list of tokens if `as_list` is `True`, as a 
        string otherwise
    """
    comment = comment.lower()
    for regex in REGEXES:
        comment = regex(comment)
    comment.strip()

    # Filtering out stop words from the `tmp` list
    tokens = [token for token in comment.split() if len(token) > 0]

    if nlp is not None:
        tokens = [ token for token in tokens if not token in nlp.Defaults.stop_words ]
    
    return ' '.join(tokens) if not as_list else tokens


def xml_to_json(xml_file: Path, json_file: Path):
    """
    Convert an `.xml` file to a `.json` file given the following tree structure:
        <comments>
            <comment>
                <movie>       {str}   </movie>
                <review_id>   {str}   </review_id>
                <name>        {str}   </name>
                <user_id>     {str}   </user_id>
                <note>        {float} </note>
                <commentaire> {str}   </commentaire>
            </comment>
            ...
         </comments>
    
    Parameter
    ---------
    xml_file: Path
        Path to `.xml` file to parse
    json_file: Path
        Path to `.json` file to write
    """
    tree = xml.etree.ElementTree.parse(xml_file)
    root = tree.getroot()
    comments = []
    tag_ops = {
        "review_id": lambda x: str(x),
        "name": lambda x: str(x),
        "user_id": lambda x: str(x),
        "commentaire": lambda x: str(x),
        "movie": lambda x: {"id": str(x)},
#         "note": lambda x: numpy.float(x.replace(",", ".")),
    }

    for idx, child in enumerate(root):
        comment = {tag: op(child.find(tag).text) for tag, op in tag_ops.items()}
        comments += [comment]

    with open(json_file, 'w', encoding='utf8') as file:
        json.dump(comments, file, ensure_ascii=False)


def parse_metadata(metadata: dict) -> dict:
    """
    Parses a film's metadata extracted from the web scrapper depicted in 
    `scrapper/`.

    Parameter
    ---------
    metadata: dict
        Film's metadata scrapped from allocine.fr

    Returns
    -------
    dict
        The parsed metadata as an objet with fields:
        {
            'title': str,
            'category': list(str),
            'avg_note': float
        }
    """
    obj = {}

    obj['title'] = metadata['title']
    obj['category'] = metadata['category']

    is_users = False

    for content in metadata['data']:
        if content.strip().lower() == 'spectateurs':
            is_user = True
        else:
            try:
                content = float(content.strip().replace(',', '.'))
                if not math.isnan(content) and is_user:
                    obj['avg_note'] = content
                    break
            except ValueError:
                continue


def json_to_ndjson(
    json_file: Path,
    out_folder: Path,
    limit: int = 200_000,
    es_index: str = 'movie_db_sw',
    movie_additional: Path = None,
):
    """
    Get the raw list of comments form a `.json` source, tokenize the 
    `commentaire` section and register the generated list into a `.ndjson` 
    file, under the `ndjson`. This function splits the data into multiple 
    files if the number of `comment` objects exceeds a given limit.
    
    Parameters
    ----------
    json_file: Path
        Original `.json` file to parse with untokenized comment.
    movie_additional: Path, optional (default: None)
        Path to additional data on movies as a `.json`file. This aims at adding 
        some extra information on movie reviews from the raw `.xml` dataset.
    out_folder: Path
        Elastic search folder to register the list of tokenized comments.
    limit: int, optional (default: 200_000)
        Maximum number of comments per file in the output pipeline.
    es_index: str, optional (default: "movie_db")
        Name of the Elastic Search database.
    """
    basename = json_file.stem

    # Creates folder given file's basename where all output files should be
    # saved an update the output path.
    pathlib.Path(out_folder / f'{basename}').mkdir(parents=True, exist_ok=True)
    out_folder = out_folder / f'{basename}'

    if out_folder.is_dir():
        clear_dir(out_folder)

    es_idx = {'index': {'_index': es_index}}

    # Loading raw json file
    with open(json_file, 'r', encoding='utf8') as file:
        raw_dataset = json.load(file)

    # Loading complementary data from web scrapping
    if movie_additional is not None:
        with open(movie_additional, 'r', encoding='utf8') as file:
            extra_info = json.load(file)

    # Tokenization and comments registration
    num_part = 0
    num_elts = len(raw_dataset)
    num_digits = len(f'{num_elts:_d}')

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

    # Clears first file to edit, in case of previous data registration
    open(out_folder / f'{basename}_{num_part}.ndjson', 'w').close()

    for idx, review in enumerate(raw_dataset):
        review['lst_mots'] = [{'text': word} for word in filter_comment(review['commentaire'], nlp=nlp, as_list=True)]
        review['num_chars'] = sum(len(token) for token in review['lst_mots'])

        if extra_info is not None:
            review['movie'].update(parse_metadata(extra_info[review['movie']['id']]))

        if idx % limit == limit - 1:
            # The maximum number of entries has been reached, we need to
            # register the next comments to a new file
            num_part += 1
            open(out_folder / f'{basename}_{num_part}.ndjson', 'w').close()

        # Registers data into the current `.ndjson` output file
        with open(out_folder / f'{basename}_{num_part}.ndjson', 'a+') as file:
            file.write(f'{json.dumps(es_idx, ensure_ascii=False)}\n')
            file.write(f'{json.dumps(review, ensure_ascii=False)}\n')


def clear_dir(folder: Path):
    for path in folder.glob('**/*'):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def xml_to_ndjson(
    xml_file: Path, json_folder: Path, es_folder: Path, movie_additional: Path = None
):
    """
    Parses a raw xml file containing the list of all movie reviews and convert 
    it to a `.ndjson` file, with  extra information on reviews if additional
    data are available.
    
    Parameters
    ----------
    xml_file: Path
        Path to the `.xml` file to parse.
    json_folder: Path
        Path to the `.json` folder that should contain the intermediate parsing
        results.
    es_folder: Path
        Path to the elastic search folder that should contain the parsed result
    movie_additional: Path
        Path to the file containing the extra data on movies shown in the 
        dataset.
    """
    basename = xml_file.stem

    # Name of the intermediate `.json` file
    json_file = json_folder / f"{basename}.json"

    xml_to_json(xml_file, json_file)

    # Parse raw `.json` movie reviews (i.e. directly extracted from the `.xml`
    # movie reviews) file into a `.ndjson` file.
    json_to_ndjson(
        json_file, es_folder, movie_additional=movie_additional, limit=10_000
    )
