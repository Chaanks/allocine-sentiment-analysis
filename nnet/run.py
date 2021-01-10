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
from tensorflow.keras import backend as K

from models import cnn

os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 128
max_features = 10000
embedding_dim = 300
sequence_length = 200
epochs = 5

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
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

    # logger.info('Number of batches in raw_ds_train: %d'
    #     % tf.data.experimental.cardinality(raw_ds_train))

    # logger.info('Number of batches in raw_ds_eval: %d'
    #     % tf.data.experimental.cardinality(raw_ds_eval))

    vectorize_layer = TextVectorization(
        standardize=None,
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
        ngrams=2
    )

    print('debug')

    # Create
    text_ds = raw_ds_train.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    # Load pre-trained
    #vl_path = root / 'data/models/model_minpad/tv_layer.pkl'
    #from_disk = pickle.load(open(vl_path, "rb"))
    #vectorize_layer = TextVectorization.from_config(from_disk['config'])
    # You have to call `adapt` with some dummy data (BUG in Keras)
    #new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"])) 
    #vectorize_layer.set_weights(from_disk['weights'])

    # Import fasttext
    fasttext = root / 'data/embedding/cc.fr.300.vec'
    embeddings_index = {}
    with open(fasttext) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    voc = vectorize_layer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    num_tokens = len(voc) + 2
    embedding_dim = 300
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    # Vectorize the data.
    ds_train = raw_ds_train.map(vectorize_text)
    ds_eval = raw_ds_eval.map(vectorize_text)

    # Do async prefetching / buffering of the data for best performance on GPU.
    ds_train = ds_train.cache().prefetch(buffer_size=10)
    ds_eval = ds_eval.cache().prefetch(buffer_size=10)

    # Fit the model using the train and test datasets.
    cfg = {'max_features': max_features, 'embedding_dim': embedding_dim, 'embedding_matrix': embedding_matrix, 'trainable': False}
    model = cnn.CovNet1D(cfg)
    #model = cnn.HomeMade(cfg)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", f1_m])
    model.fit(ds_train, validation_data=ds_eval, epochs=epochs)

    # Save model and vectorizer
    model.save('../data/models/model_v2')
    pickle.dump({'config': vectorize_layer.get_config(),
                'weights': vectorize_layer.get_weights()}
                , open("../data/models/model_v2/tv_layer.pkl", "wb"))
