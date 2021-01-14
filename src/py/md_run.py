#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""md_run.py: 
    Last minute script used to develop and train a fined tuned model.
"""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "2.0.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]
__license__ = "MIT"

import os
import pickle
import pathlib
from pathlib import Path
import numpy as np
from loguru import logger
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import yaml
import utils
from model.mlpcnn import create_model
import const
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

max_features = 20000
embedding_dim = 200
sequence_length = 1000

LBL_TO_IDX = {
    "0.5": 0,
    "1.0": 1,
    "1.5": 2,
    "2.0": 3,
    "2.5": 4,
    "3.0": 5,
    "3.5": 6,
    "4.0": 7,
    "4.5": 8,
    "5.0": 9,
}


def feats_encoding(df):
    # encode numerical variables
    inputs = {}
    for name, column in df.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32

        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    numeric_inputs = {
        name: input for name, input in inputs.items() if input.dtype == tf.float32
    }

    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = preprocessing.Normalization()
    norm.adapt(np.array(df[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)

    preprocessed_inputs = [all_numeric_inputs]  # all_numeric_inputs

    # encode categorial variables
    for feature in ["directors", "kinds"]:  #'movie_id',
        lookup = preprocessing.StringLookup(vocabulary=np.unique(df[feature]))
        one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())

        x = lookup(inputs[feature])
        x = one_hot(x)
        preprocessed_inputs.append(x)

    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
    return tf.keras.Model(inputs, preprocessed_inputs_cat), inputs


def reviews_encoding(df, max_features, sequence_length):
    vectorize_layer = TextVectorization(
        standardize=None,
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    # vectorize_layer.adapt(np.array(df['commentaire']))
    vectorize_layer.adapt(np.array(df))

    return vectorize_layer


def dataprep(df):
    # split dataframe
    df_feats = df.copy().drop(
        columns=["directors", "kinds", "review_id", "note_mean", "note_std"]
    )  # "rating", "review" "subscriber", "movie_id", "note_mean", "note_std"
    df_reviews = df_feats.pop("commentaire")
    df_labels = df_feats.pop("note")

    # df_labels = df_labels.astype({"note": np.float32})

    df_labels = df_labels.tolist()
    df_labels_enc = []
    for label in df_labels:
        df_labels_enc.append(LBL_TO_IDX[str(label)])
    df_labels = tf.keras.utils.to_categorical(df_labels_enc, 10)

    return df_feats, df_reviews, df_labels


if __name__ == "__main__":
    root = Path("/home/jarod/git/innovation/allocine-sentiment-analysis")
    train_path = root / "data/csv/train_std.csv"
    dev_path = root / "data/csv/dev_std.csv"
    dtype = {
        "note": np.float32,
        "movie_id": str,
        "rating": np.float32,
        "review": np.float32,
        "subscriber": np.float32,
    }

    df_train = pd.read_csv(train_path, dtype=dtype)
    df_dev = pd.read_csv(dev_path, dtype=dtype)

    train_feats, train_reviews, train_labels = dataprep(df_train)
    dev_feats, dev_reviews, dev_labels = dataprep(df_dev)

    train_feats_dict = {name: np.array(value) for name, value in train_feats.items()}
    train_feats_dict = {name: values for name, values in train_feats_dict.items()}

    dev_feats_dict = {name: np.array(value) for name, value in dev_feats.items()}
    dev_feats_dict = {name: values for name, values in dev_feats_dict.items()}

    feats_encoder, inputs = feats_encoding(train_feats)
    reviews_encoder = reviews_encoding(train_reviews, max_features, sequence_length)

    model = create_model(
        feats_encoder, inputs, reviews_encoder, max_features, embedding_dim
    )
    model.compile(
        loss="categorical_crossentropy", optimizer="nadam", metrics="accuracy"
    )

    model.fit(
        x=[train_feats_dict, train_reviews],
        y=train_labels,
        validation_data=([dev_feats_dict, dev_reviews], dev_labels),
        epochs=3,
        batch_size=64,
    )

    # Save configuration, model and embeddings in `out/` folder
    model_dir = root / "data/model/md"
    model.save(model_dir / "model")
    pickle.dump(
        {"cfg": feats_encoder.get_config(), "w": feats_encoder.get_weights()},
        open(model_dir / "tv_layer.pkl", "wb"),
    )

    # print(model.summary())

    test_path = root / "data/csv/test_std_clean.csv"
    df_test = pd.read_csv(test_path, dtype=dtype)

    test_feats = df_test.copy().drop(
        columns=["directors", "kinds", "note_mean", "note_std"]
    )  # "rating", "review",	"subscriber", "movie_id", "note_mean", "note_std"
    test_reviews = test_feats.pop("commentaire")
    id_lst = test_feats.pop("review_id")

    test_feats_dict = {name: np.array(value) for name, value in test_feats.items()}
    test_feats_dict = {name: values for name, values in test_feats_dict.items()}

    results = model.predict(x=[test_feats_dict, test_reviews])

    preds = []
    for i, p in enumerate(tqdm(results)):
        m = int(tf.math.argmax(p))
        idx = const.IDX_TO_LBL[m]
        preds.append((id_lst[i], idx))

    with open(model_dir / "score.txt", "w") as file:
        for review_id, pred in preds:
            file.write(f"{review_id} {pred}\n")
