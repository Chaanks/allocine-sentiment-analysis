#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run.py: This file is used as a launcher to train our model.
"""

__author__ = 'Jonathan Heno, Duret Jarod'
__license__ = 'MIT'

import os
import pickle
from pathlib import Path
import numpy as np
from loguru import logger
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from models.cnn import CovNet1D

os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 256
max_features = 20000
embedding_dim = 128
sequence_length = 10000
epochs = 5

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

if __name__ == '__main__':

    #tf.config.experimental.list_physical_devices('CPU'))

    root = Path('/home/jarod/git/allocine-sentiment-analysis')
    corpus = root / 'data/allocine_filtered'
    train = corpus / 'train'
    eval = corpus / 'eval'

    # Create dataset
    raw_ds_train = tf.keras.preprocessing.text_dataset_from_directory(
    train,
    label_mode='categorical',
    batch_size=BATCH_SIZE
    )

    raw_ds_eval = tf.keras.preprocessing.text_dataset_from_directory(
        eval,
        label_mode='categorical',
        batch_size=BATCH_SIZE
    )

    logger.info('Number of batches in raw_ds_train: %d'
        % tf.data.experimental.cardinality(raw_ds_train))

    logger.info('Number of batches in raw_ds_eval: %d'
        % tf.data.experimental.cardinality(raw_ds_eval))

    vectorize_layer = TextVectorization(
        standardize=None,
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    # Create
    text_ds = raw_ds_train.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    # Load pre-trained
    #vl_path = root / 'data/models/model_final/tv_layer.pkl'
    #from_disk = pickle.load(open(vl_path, "rb"))
    #vectorize_layer = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    #new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"])) 
    #vectorize_layer.set_weights(from_disk['weights'])

    # Vectorize the data.
    ds_train = raw_ds_train.map(vectorize_text)
    ds_eval = raw_ds_eval.map(vectorize_text)

    # Do async prefetching / buffering of the data for best performance on GPU.
    ds_train = ds_train.cache().prefetch(buffer_size=10)
    ds_eval = ds_eval.cache().prefetch(buffer_size=10)

    # Fit the model using the train and test datasets.
    cfg = {'max_features': max_features, 'embedding_dim': embedding_dim}
    model = CovNet1D(cfg)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(ds_train, validation_data=ds_eval, epochs=epochs)

    # Save model and vectorizer
    model.save('../data/models/model_baseline')
    pickle.dump({'config': vectorize_layer.get_config(),
                'weights': vectorize_layer.get_weights()}
                , open("../data/models/model_baseline/tv_layer.pkl", "wb"))