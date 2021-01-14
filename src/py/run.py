#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""run.py: This file is used as a launcher to train our model."""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.5.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]
__license__ = "MIT"


import argparse
import numpy as np
import os
import pickle
import tensorflow
import yaml

import const
import parser
import utils

from loguru import logger
from pathlib import Path
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from model import conv_seq, conv_par


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def vectorize(text: str, label: float, vlayer: TextVectorization):
    """
    Transforms text into floating tensor with its rating.

    Parameters
    ----------
    text (tensorflow.python.framework.ops.Tensor):
        Review to vectorize.
    label (tensorflow.python.framework.ops.Tensor):
        Rating linked with this text.
    vlayer (tensorflow.python.keras.layers.preprocessing.text_vectorization.TextVectorization):
        Text vectorization model

    Returns
    -------
    (tensor, float): The corresponding floating tensor with its rating.
    """
    text = tensorflow.expand_dims(text, -1)
    return vlayer(text), label


def build_model(mtype: str, cfg: argparse.Namespace) -> conv_par.ConvPar or conv_seq.ConvSeq:
    """
    Builds the model given a model type and a configuration.

    Parameters
    ----------
    mtype (str):
        Type of model to instanciate.
    cfg (argparse.Namespace):
        The configuration of the model.

    Returns
    -------
    (conv_par.ConvPar or conv_seq.ConvSeq):
        The model.

    Raises
    ------
    (AssertionError):
        If the configuration file given for the model does not contain the 
        appropriate structure.
    """
    assert 'emb_dim' in cfg['corpus'] and 'voc_len' in cfg['corpus'], const.WRONG_CORPUS_DEFINITION

    if mtype == const.MODEL_PAR:
        assert 'num_filters' in cfg['model'] and 'convs' in cfg['model'] and 'dropout' in cfg['model'], const.WRONG_CONFIG_FILE(mtype)
        return conv_par.ConvPar(
            len(const.IDX_TO_LBL),
            cfg["corpus"]["voc_len"],
            cfg["corpus"]["emb_dim"],
            cfg["model"]["dropout"],
            cfg["model"]["num_filters"],
            cfg["model"]["convs"],
            cfg["model"]["fc_dim"],
        )
    else:
        assert 'layers' in cfg['model'], const.WRONG_CONFIG_FILE(mtype)
        return conv_seq.ConvSeq(
            len(const.IDX_TO_LBL),
            cfg["corpus"]["voc_len"],
            cfg["corpus"]["emb_dim"],
            cfg["model"]["layers"],
        )


if __name__ == "__main__":
    args = parser.parse_args(const.TRAIN_MODE)

    assert args.config

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    out = Path(args.out)
    if out.is_dir():
        utils.clear_dir(out)
    else:
        out.mkdir(parents=True, exist_ok=True)

    train = Path(args.train)
    dev = Path(args.dev)

    # tensorflow.config.experimental.list_physical_devices('CPU'))

    # Instanciating word embedding model
    vlayer = TextVectorization(
        standardize=None,
        max_tokens=cfg["corpus"]["voc_len"],
        output_mode="int",
        output_sequence_length=cfg["corpus"]["seq_len"],
    )

    # Retrieving datasets
    raw_train_ds = tensorflow.keras.preprocessing.text_dataset_from_directory(
        train, label_mode="categorical", batch_size=cfg["exp"]["batch_size"]
    )
    logger.info(
        "Number of batches in train set: %d"
        % tensorflow.data.experimental.cardinality(raw_train_ds)
    )
    reviews = raw_train_ds.map(lambda x, y: x)
    vlayer.adapt(reviews)
    # Generating embeddings
    train_ds = raw_train_ds.map(lambda rev, lbl: vectorize(rev, lbl, vlayer))
    # Do async prefetching / buffering of the data for best performance on GPU
    train_ds = train_ds.cache().prefetch(buffer_size=10)

    if args.dev:
        raw_dev_ds = tensorflow.keras.preprocessing.text_dataset_from_directory(
            dev, label_mode="categorical", batch_size=cfg["exp"]["batch_size"]
        )
        logger.info(
            "Number of batches in dev set: %d"
            % tensorflow.data.experimental.cardinality(raw_dev_ds)
        )
        dev_ds = raw_dev_ds.map(lambda rev, lbl: vectorize(rev, lbl, vlayer))
        dev_ds = dev_ds.cache().prefetch(buffer_size=10)

    # Fit the model using the train and test datasets.
    model = build_model(args.model, cfg)
    model.compile(
        loss=cfg["model"]["loss"],
        optimizer=cfg["model"]["optimizer"],
        metrics=cfg["model"]["metrics"],
    )

    csv_logger = tensorflow.keras.callbacks.CSVLogger(
        out / const.TRAIN_LOG_OUTPUT_FILE, separator="\t"
    )

    if args.dev:
        model.fit(
            train_ds,
            validation_data=dev_ds,
            epochs=cfg["exp"]["epochs"],
            callbacks=[csv_logger],
        )
    else:
        model.fit(
            train_ds,
            epochs=cfg["exp"]["epochs"],
            callbacks=[csv_logger],
        )

    # Save configuration, model and embeddings in `out/` folder
    with open(out / const.CFG_OUTPUT_FILE, "w") as file:
        yaml.safe_dump(cfg, file, indent=2)
    model.save(out / "model")
    pickle.dump(
        {"cfg": vlayer.get_config(), "w": vlayer.get_weights()},
        open(out / const.VLAYER_OUTPUT_FILE, "wb"),
    )
