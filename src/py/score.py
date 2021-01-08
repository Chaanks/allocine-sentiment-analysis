#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""scoring.py: This file is used to test a trained model."""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.0.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]
__license__ = "MIT"


import numpy as np
import os
import pandas as pd
import pickle
import string
import tensorflow as tf
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
    test = Path(args.test)

    # Load pre-trained model
    vl_path = mod_dir / const.VLAYER_OUTPUT_FILE
    vl = pickle.load(open(vl_path, "rb"))
    vlayer = TextVectorization.from_config(vl["config"])
    vlayer.set_weights(vl["weights"])
    # [BUG KERAS] You have to call `adapt` with some dummy data
    # new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))

    model = tf.keras.models.load_model(mod_dir)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # A string input
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    # Turn strings into vocab indices
    indices = vlayer(inputs)
    # Turn vocab indices into predictions
    outputs = model(indices)

    # Our end to end model
    end_to_end_model = tf.keras.Model(inputs, outputs)
    end_to_end_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    trials = pd.read_csv(test)

    preds = []

    for index, row in trials.iterrows():
        text = tf.expand_dims(row["commentaire"], -1)
        result = end_to_end_model.predict(text)
        idx = const.IDX_TO_LBL[tf.math.argmax(result)]

    text = tf.expand_dims("mauvais film", -1)
    result = end_to_end_model.predict(text)
    print(result)
