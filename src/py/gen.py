#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""gen.py: Dataset generation."""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.5.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]
__license__ = "MIT"


import json
import numpy
import pandas
import pathlib
import re
import spacy
import xml.etree.ElementTree

import const
import parser
import utils

from loguru import logger
from tqdm import tqdm


def xml_to_json(xml_file: pathlib.Path, json_file: pathlib.Path):
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
    
    Parameters
    ----------
    xml_file: pathlib.Path
        Path to `.xml` file to parse.
    json_file: pathlib.Path
        Path to `.json` file to write.
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
        "note": lambda x: numpy.float(x.replace(",", ".")),
    }

    for idx, child in enumerate(tqdm(root)):
        comment = {
            tag: op(child.find(tag).text)
            for tag, op in tag_ops.items()
            if child.find(tag) != None
        }
        comments += [comment]

    with open(json_file, "w", encoding="utf8") as file:
        json.dump(comments, file, indent=2, ensure_ascii=False)


def json_to_ndjson(
    json_file: pathlib.Path,
    out_dir: pathlib.Path,
    limit: int = 200_000,
    es_index: str = "movie_db",
    movie_extra: pathlib.Path = None,
    std: bool = True,
    nlp=None,
):
    """
    Get the raw list of comments form a `.json` source, tokenize the 
    `commentaire` section and register the generated list into a `.ndjson` 
    file format. This function splits the data into multiple files if the 
    number of `comment` objects exceeds a given limit.
    
    Parameters
    ----------
    json_file: pathlib.Path
        Original `.json` file to parse with untokenized comment.
    out_dir: pathlib.Path
        Elastic search folder to register the list of tokenized comments.
    movie_extra: pathlib.Path, optional (default: None)
        Path to additional data on movies as a `.json`file. This aims at 
        adding some extra information on movie reviews from the raw `.xml` 
        dataset.
    limit: int, optional (default: 200_000)
        Maximum number of comments per file in the output pipeline.
    es_index: str, optional (default: "movie_db")
        Name of the Elastic Search database.
    std: bool (default: True)
        If True this method will standardize the textual content with preset.
    nlp: NLP SpaCy model (default: None)
        NLP model to load while parsing the raw dataset. This extra utility 
        helps us to filter out stop words during reviews' preprocessing.
    """
    basename = json_file.stem

    # Creates folder given file's basename where all output files should be
    # saved an update the output path.
    out_dir = out_dir / basename
    out_dir.mkdir(parents=True, exist_ok=True)

    es_idx = {"index": {"_index": es_index}}

    # Loading raw json file
    with open(json_file, "r", encoding="utf8") as file:
        raw_dataset = json.load(file)

    # Loading complementary data from web scrapping
    if movie_extra is not None:
        with open(movie_extra, "r", encoding="utf8") as file:
            movie_extra = json.load(file)

    # Tokenization and comments registration
    num_part = 0
    num_elts = len(raw_dataset)
    num_digits = len(f"{num_elts:_d}")

    # Clears first file to edit, in case of previous data registration
    open(out_dir / f"{basename}_{num_part}.ndjson", "w").close()

    for idx, review in enumerate(tqdm(raw_dataset)):
        review["lst_mots"] = [
            {"text": word}
            for word in utils.filter_comment(
                review["commentaire"], nlp=nlp, std=std, as_list=True
            )
        ]
        review["num_chars"] = sum(len(token) for token in review["lst_mots"])

        if movie_extra is not None:
            review["movie"].update(
                utils.parse_metadata(movie_extra[review["movie"]["id"]])
            )

        if idx % limit == limit - 1:
            # The maximum number of entries has been reached, we need to
            # register the next comments to a new file
            num_part += 1
            open(out_dir / f"{basename}_{num_part}.ndjson", "w").close()

        # Registers data into the current `.ndjson` output file
        with open(out_dir / f"{basename}_{num_part}.ndjson", "a+") as file:
            file.write(f"{json.dumps(es_idx, ensure_ascii=False)}\n")
            file.write(f"{json.dumps(review, ensure_ascii=False)}\n")


def gen_dataset(
    ds_file: pathlib.Path, out_dir: pathlib.Path, std: bool = True, nlp=None
):
    """
    Extracts a dataset from a `.json` formatted dataset and stores it under a 
    keras readable architecture.

    Parameters
    ----------
    ds_file: pathlib.Path
        Path to .json dataset.
    out_dir: pathlib.Path
        Output directory that should contain the whole dataset architecture
        in conformity with keras standards.
    std: bool (default: True)
        If True this method will standardize the textual content with preset.
    nlp: NLP SpaCy model (default: None)
        NLP model to load while parsing the raw dataset. This extra utility 
        helps us to filter out stop words during reviews' preprocessing.
    """
    # Creates or clears dataset folder
    ds_dir = out_dir / ds_file.stem
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Loading raw json file
    with open(ds_file, "r", encoding="utf8") as file:
        ds = json.load(file)

    # Write files
    for review in tqdm(ds):
        # Filters and standardize review's content
        filtered = utils.filter_comment(review["commentaire"], std=std, nlp=nlp)
        if len(filtered) == 0:
            logger.warning(f"Review {review['review_id']} filtered out")
            continue

        # Registers comment to its corresponding label folder
        label_dir = ds_dir / str(review["note"])
        label_dir.mkdir(parents=True, exist_ok=True)

        out_file = label_dir / f"{review['review_id'].replace('review_', '')}.txt"
        out_file.touch()

        with open(out_file, "w") as file:
            file.write(filtered)


def gen_trials(
    ds_file: pathlib.Path, out_dir: pathlib.Path, std: bool = True, nlp=None
):
    """
    Extracts a dataset from a `.json` formatted dataset and stores it under a 
    keras readable architecture.

    Parameters
    ----------
    ds_file: pathlib.Path
        Path to .json dataset.
    out_dir: pathlib.Path
        Output directory that should contain the whole dataset architecture
        in conformity with keras standards.
    std: bool (default: True)
        If True this method will standardize the textual content with preset.
    nlp: NLP SpaCy model (default: None)
        NLP model to load while parsing the raw dataset. This extra utility 
        helps us to filter out stop words during reviews' preprocessing.
    """
    # Loading raw json file
    with open(ds_file, "r", encoding="utf8") as file:
        ds = json.load(file)

    reviews = {"review_id": [], "commentaire": []}

    for review in tqdm(ds):
        # Filters and standardize review's content
        filtered = utils.filter_comment(review["commentaire"], nlp=nlp, std=std)

        # If the filtered content is empty, we notify the user and setup the
        # entry of this trial back to the original comment
        if len(filtered) == 0:
            logger.warning(f"Review {review['review_id']} is empty")
            filtered = review["commentaire"]

        reviews["review_id"].append(review["review_id"])
        reviews["commentaire"].append(filtered)

    pandas.DataFrame.from_dict(reviews).to_csv(out_dir / const.TRIALS_FILENAME)


if __name__ == "__main__":
    args = parser.parse_args(const.GEN_MODE)

    if args.stopwords:
        # Loading SpaCy model from tokenization utilities
        spacy.prefer_gpu()
        nlp = spacy.load("fr_core_news_sm")

        with open(pathlib.Path(args.stopwords), "r", encoding="utf8") as file:
            nlp.Defaults.stop_words = set(json.load(file))
    else:
        nlp = None

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.in_format == const.RAW_FORMAT:
        for path in args.data:
            p = pathlib.Path(path)

            json_file = out / f"{p.stem}.json"
            xml_to_json(p, json_file)

            if args.out_format == const.ES_FORMAT:
                json_to_ndjson(
                    json_file,
                    out,
                    limit=10_000,
                    movie_extra=pathlib.Path(args.extra) if args.extra else None,
                    es_idx=args.es_idx,
                    std=args.standardization,
                    nlp=nlp,
                )
            elif args.out_format == const.TRAIN_FORMAT:
                gen_dataset(json_file, out, nlp=nlp, std=args.standardization)
            elif args.out_format == const.TRIALS_FORMAT:
                gen_trials(json_file, out, nlp=nlp, std=args.standardization)

            if args.out_format != const.JSON_FORMAT:
                json_file.unlink()
    elif args.in_format == const.JSON_FORMAT:
        for path in args.data:
            json_file = pathlib.Path(path)

            if args.out_format == const.ES_FORMAT:
                json_to_ndjson(
                    json_file,
                    out,
                    limit=10_000,
                    movie_extra=pathlib.Path(args.extra) if args.extra else None,
                    std=args.standardization,
                    nlp=nlp,
                )
            elif args.out_format == const.TRAIN_FORMAT:
                gen_dataset(json_file, json, out, std=args.standardization, nlp=nlp)
            elif args.out_format == const.TRIALS_FORMAT:
                gen_trials(json_file, out, std=args.standardization, nlp=nlp)
