#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run.py: This file is used as a launcher to train our model.
"""

__author__ = 'Jonathan Heno, Duret Jarod'
__license__ = 'MIT'

import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import string
import pickle
import pandas as pd
from loguru import logger

from models.cnn import CovNet1D

os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 128
max_features = 20000
embedding_dim = 128
sequence_length = 2000

IDX2LABEL = {
    0: '0,5',
    1: '1,0',
    2: '1,5',
    3: '2,0',
    4: '2,5',
    5: '3,0',
    6: '3,5',
    7: '4,0',
    8: '4,5',
    9: '5,0',
}

if __name__ == '__main__':

    root = Path('/home/jarod/git/allocine-sentiment-analysis')
    corpus = root / 'data/allocine_filtered'
    test = corpus / 'test'

    # Load pre-trained
    vl_path = root / 'data/models/model_baseline/tv_layer.pkl'
    from_disk = pickle.load(open(vl_path, "rb"))
    vectorize_layer = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    #new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"])) 
    vectorize_layer.set_weights(from_disk['weights'])

    cfg = {'max_features': max_features, 'embedding_dim': embedding_dim}
    model = tf.keras.models.load_model('../data/models/model_baseline')
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # A string input
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    # Turn strings into vocab indices
    indices = vectorize_layer(inputs)
    # Turn vocab indices into predictions
    outputs = model(indices)

    # Our end to end model
    end_to_end_model = tf.keras.Model(inputs, outputs)
    end_to_end_model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    df = pd.read_csv('../data/allocine_filtered/test/trials.csv')
    
    preds = []
    
    for index, row in df.iterrows():
        text = tf.expand_dims(row['commentaire'], -1)
        result = end_to_end_model.predict(text)
        idx = IDX2LABEL[tf.math.argmax(result)]

    text = tf.expand_dims('mauvais film', -1)
    result = end_to_end_model.predict(text)
    print(result)