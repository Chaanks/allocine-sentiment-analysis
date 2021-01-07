#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""const.py: Gathers all the constants used in the projects"""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.0.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = ["jarod.duret@alumni.univ-avignon.fr", "jonathan.heno@alumni.univ-avignon.fr"]
__license__ = "MIT"


# Parsing
REGEXES = [
    lambda x: re.sub(r"http\S+", " ", x),
    lambda x: re.sub(
        r"[^\d|(a-z)|\U0001F600-\U0001F64F|!|.|#|@|è|é|à|ù|ô|ü|ë|ä|û|î|ê|â|ç\s]", " ", x
    ),
    lambda x: re.sub(r"<br />", " ", x),
    lambda x: re.sub(r"[\(|\)]", " ", x),
    # lambda x: re.sub(r'(.+?)\1{2,}', r'\1\1\1', x),
    lambda x: re.sub(r"\s+", " ", x),
]

RUNNING_MODES = ['train', 'score']
TRAIN_MODE = 'train'
SCORE_MODE = 'score'


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

VLAYER_OUTPUT_FILE = "tv_layer.pkl"


# Errors
UNKNOWN_RUNNING_MODE = lambda mode: f"Unknown {mode} mode ({'run'|'score'} expected)."
NO_RUNNING_MODE = "No running mode given ({'run'|'score'} expected)."