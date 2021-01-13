#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""const.py: Gathers all the constants used in the projects"""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.5.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]
__license__ = "MIT"


import re


# Parsing
REGEXES = [
    lambda x: re.sub(r"http\S+", " ", x),
    lambda x: re.sub(
        r"[^\d|(a-z)|\U0001F600-\U0001F64F|!|.|#|@|è|é|à|ù|ô|ü|ë|ä|û|î|ê|â|ç\s]", " ", x
    ),
    lambda x: re.sub(r"<br />", " ", x),
    lambda x: re.sub(r"[\(|\)]", " ", x),
    lambda x: re.sub(r"(.+?)\1{2,}", r"\1\1\1", x),
    lambda x: re.sub(r"\s+", " ", x),
]

# User input
TRAIN_MODE = "train"
SCORE_MODE = "score"
GEN_MODE = "gen"
RUNNING_MODES = [TRAIN_MODE, SCORE_MODE, GEN_MODE]

RAW_FORMAT = "raw"
JSON_FORMAT = "json"
ES_FORMAT = "es"
TRIALS_FORMAT = "score"
TRAIN_FORMAT = "run"
IN_FORMATS = [RAW_FORMAT, JSON_FORMAT]
OUT_FORMATS = [JSON_FORMAT, ES_FORMAT, TRIALS_FORMAT, TRAIN_FORMAT]

TRIALS_FILENAME = "trials.csv"

MODEL_PAR = "conv1dpar"
MODEL_SEQ = "conv1dseq"
MODELS = [MODEL_PAR, MODEL_SEQ]

# Model
IDX_TO_LBL = {
    0: "0,5",
    1: "1,0",
    2: "1,5",
    3: "2,0",
    4: "2,5",
    5: "3,0",
    6: "3,5",
    7: "4,0",
    8: "4,5",
    9: "5,0",
}

CFG_OUTPUT_FILE = "cfg.yml"
VLAYER_OUTPUT_FILE = "tv_layer.pkl"
TRAIN_LOG_OUTPUT_FILE = "logs.csv"


# Errors
UNKNOWN_RUNNING_MODE = (
    lambda mode: f"\033[1;33mUnknown {mode} mode ({{'run'|'score'|'gen'}} expected).\033[0m"
)
NO_RUNNING_MODE = (
    "\033[1;33mNo running mode given ({'run'|'score'|'gen'} expected).\033[0m"
)
OUT_IS_NOT_DIRECTORY = (
    lambda path: f"\033[1;33mOutput path {path} is not a directory.\033[0m"
)
SAME_IO_FORMAT = "\033[1;33mSame input and output format given.\033[0m"
EXTRA_WITHOUT_ES_OUTPUT = (
    f"\033[1;33mGave path to extra movie metadata set without {ES_FORMAT}.\033[0m"
)
ES_IDX_WITHOUT_ES_OUTPUT = f"\033[1;33mGave ES index without {ES_FORMAT}.\033[0m"
TOO_MANY_OPTIONS_PROVIDED_FOR_JSON_OUTPUT = (
    f"\033[1;33mToo many arguments given for {JSON_FORMAT} output.\033[0m"
)
UNKNOWN_LAYER_TYPE = lambda m: f"\033[1;33mUnknown layer for {m}.\033[0m"
NO_STD_FOR_JSON_FORMAT = f"""\033[1;33m\
Review standardization is not an available option for {JSON_FORMAT} output.\
\033[0m"""
WRONG_CONFIG_FILE = lambda model: f"\033[1;33mWrong configuration file for {model}.\033[0m"
