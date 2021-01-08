#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""run.py: This file is used as a launcher to train our model."""

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
import pickle
import tensorflow as tf
import yaml

import const
import parser
import utils

from loguru import logger
from model.cnn_baseline import CNNBaseline
from pathlib import Path
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def vectorize(text, label):
    """Transforms text into floating tensor with its rating.

    Parameters
    ----------
        text: str
            Review to vectorize.
        label: float
            Rating linked with this text.
    
    Returns
    -------
        (tensor, float): The corresponding floating tensor with its rating.
    """
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label  # ?????


if __name__ == "__main__":
    args = parser.parse_args(const.TRAIN_MODE)

    assert args.config

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)
    print(cfg)

    out = Path(args.out)
    if out.is_dir():
        utils.clear_dir(out)

    train = Path(args.train)
    dev = Path(args.dev)

    # tf.config.experimental.list_physical_devices('CPU'))

    # Retrieving dataset
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train, label_mode="categorical", batch_size=cfg["batch_size"]
    )
    raw_dev_ds = tf.keras.preprocessing.text_dataset_from_directory(
        dev, label_mode="categorical", batch_size=cfg["batch_size"]
    )
    logger.info(
        "Number of batches in train set: %d"
        % tf.data.experimental.cardinality(raw_train_ds)
    )
    logger.info(
        "Number of batches in dev set: %d"
        % tf.data.experimental.cardinality(raw_dev_ds)
    )

    # Instanciating word embedding model
    vlayer = TextVectorization(
        standardize=None,
        max_tokens=cfg["voc_len"],
        output_mode="int",
        output_sequence_length=cfg["seq_len"],
    )
    reviews = raw_train_ds.map(lambda x, y: x)
    vlayer.adapt(reviews)

    # Generating embeddings
    train_ds = raw_train_ds.map(vectorize)
    dev_ds = raw_dev_ds.map(vectorize)

    # Do async prefetching / buffering of the data for best performance on GPU
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    dev_ds = dev_ds.cache().prefetch(buffer_size=10)

    # Fit the model using the train and test datasets.
    model = CNNBaseline(len(const.IDX_TO_LBL), **cfg["model"])
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=dev_ds, epochs=cfg["exp"]["epochs"])

    # Save configuration, model and embeddings in `out/` folder
    with open(out / const.CFG_OUTPUT_FILE, "w") as file:
        yaml.safe_dump(cfg, file, indent=2)
    model.save(out / "model")
    pickle.dump(
        {"cfg": vlayer.get_config(), "weights": vlayer.get_weights()},
        open(out / const.VLAYER_OUTPUT_FILE, "wb"),
    )
