#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""utils.py:
    Utilitary functions used to parse and process the original raw dataset.
"""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.5.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]
__license__ = "MIT"


import math
import pathlib
import shutil
import xml.etree.ElementTree

from pathlib import Path, PurePath
from tqdm import tqdm

import const


def filter_comment(
    comment: str, nlp=None, std: bool = True, as_list: bool = False
) -> str or list:
    """
    Filters out a comment with the list of regular expressions given in 
    `const.REGEXES`.

    Parameters
    ----------
    comment: str
        Content to filter.
    nlp: NLP SpaCy model (default: None)
        NLP model to load while parsing the raw dataset. This extra utility 
        helps us to filter out stop words during reviews' preprocessing.
    std: bool (default: True)
        If True this method will standardize the textual content with preset.
        regexes.
    as_list: bool (default: False)
        If the method should retourn a list of tokens or as a simple string.

    Returns
    -------
    str or list(str)
        The filtered comment as a list of tokens if `as_list` is `True`, as a 
        string otherwise.
    """
    comment = comment.lower()

    if std:
        for regex in const.REGEXES:
            comment = regex(comment)
        comment.strip()

    # Filtering out stop words from the `tmp` list
    tokens = [token for token in comment.split() if len(token) > 0]

    if nlp is not None:
        tokens = [token for token in tokens if not token in nlp.Defaults.stop_words]

    return " ".join(tokens) if not as_list else tokens


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

    obj["title"] = metadata["title"]
    obj["category"] = metadata["category"]

    is_users = False

    for content in metadata["data"]:
        if content.strip().lower() == "spectateurs":
            is_user = True
        else:
            try:
                content = float(content.strip().replace(",", "."))
                if not math.isnan(content) and is_user:
                    obj["avg_note"] = content
                    break
            except ValueError:
                continue


def clear_dir(folder: Path):
    for path in folder.glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
