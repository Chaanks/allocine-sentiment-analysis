#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""scoring.py: This file is used to test a trained model."""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.5.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]
__license__ = "MIT"


import os
import pandas
import pickle
import string
import tensorflow
import yaml

import const
import parser

from loguru import logger
from model.cnn_baseline import CNNBaseline
from pathlib import Path
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == "__main__":
    args = parser.parse_args(const.SCORE_MODE)

    mod_dir = Path(args.model)
    trials_path = Path(args.trials)

    with open(mod_dir / const.CFG_OUTPUT_FILE, "r") as file:
        cfg = yaml.safe_load(file)

    # Load pre-trained model
    model = tensorflow.keras.models.load_model(mod_dir)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Load embedding model
    vl_path = mod_dir / const.VLAYER_OUTPUT_FILE
    vl = pickle.load(open(vl_path, "rb"))
    vlayer = TextVectorization.from_config(vl["config"])
    vlayer.set_weights(vl["weights"])
    # [BUG KERAS] You have to call `adapt` with some dummy data
    # new_v.adapt(tensorflow.data.Dataset.from_tensor_slices(["xyz"]))

    # A string input
    inputs = tensorflow.keras.Input(shape=(1,), dtype="string")
    # Turn strings into vocab indices
    indices = vlayer(inputs)
    # Turn vocab indices into predictions
    outputs = model(indices)

    # Our end to end model
    end_to_end_model = tensorflow.keras.Model(inputs, outputs)
    end_to_end_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    trials = pandas.read_csv(trials_path)

    preds = []

    for index, row in trials.iterrows():
        text = tensorflow.expand_dims(row["commentaire"], -1)
        result = end_to_end_model.predict(text)
        idx = const.IDX_TO_LBL[tensorflow.math.argmax(result)]
