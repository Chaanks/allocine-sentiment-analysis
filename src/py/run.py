#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""run.py: This file is used as a launcher to train our model."""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.0.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = ["jarod.duret@alumni.univ-avignon.fr", "jonathan.heno@alumni.univ-avignon.fr"]
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
from models.cnn import CovNet1D
from pathlib import Path
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def vectorize_text(text, label):
    """Transforms text into floating tensor.

    Parameters
    ----------
        text: <type>
            <desc>
        label: <type>
            <desc>
    Returns
    -------
        (<type>, <type>): <desc>
    """
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


if __name__ == "__main__":
    args = parser.parse_args(const.TRAIN_MODE)

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)
    print(cfg)

    out = Path(args.out)
    if out.is_dir():
        utils.clear_dir(out)

    train = Path(args.train)
    dev = Path(args.dev)

    # tf.config.experimental.list_physical_devices('CPU'))

    # root = Path("/home/jarod/git/allocine-sentiment-analysis")
    # corpus = root / "data/allocine_filtered"
    # train = corpus / "train"
    # eval = corpus / "eval"

    # Create dataset
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train, label_mode="categorical", batch_size=cfg['batch_size']
    )

    raw_dev_ds = tf.keras.preprocessing.text_dataset_from_directory(
        eval, label_mode="categorical", batch_size=cfg['batch_size']
    )

    logger.info(
        "Number of batches in raw_train_ds: %d"
        % tf.data.experimental.cardinality(raw_train_ds)
    )

    logger.info(
        "Number of batches in raw_dev_ds: %d"
        % tf.data.experimental.cardinality(raw_dev_ds)
    )

    vectorize_layer = TextVectorization(
        standardize=None,
        max_tokens=cfg['voc_len'],
        output_mode="int",
        output_sequence_length=cfg['seq_len'],
    )

    # Create
    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    # Load pre-trained
    # vl_path = root / 'data/models/model_final/tv_layer.pkl'
    # from_disk = pickle.load(open(vl_path, "rb"))
    # vectorize_layer = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    # new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    # vectorize_layer.set_weights(from_disk['weights'])

    # Vectorize the data.
    train_ds = raw_train_ds.map(vectorize_text)
    dev_ds = raw_dev_ds.map(vectorize_text)

    # Do async prefetching / buffering of the data for best performance on GPU.
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    dev_ds = dev_ds.cache().prefetch(buffer_size=10)

    # Fit the model using the train and test datasets.
    cfg = {"max_features": max_features, "embedding_dim": embedding_dim}
    model = CovNet1D(len(const.IDX_TO_LBL), **cfg)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=dev_ds, epochs=epochs)

    # Save model and vectorizer to `out` folder
    model.save(out)
    pickle.dump(
        {
            "cfg": vectorize_layer.get_config(),
            "weights": vectorize_layer.get_weights(),
        },
        open(out / const.VLAYER_OUTPUT_FILE, "wb")
    )
