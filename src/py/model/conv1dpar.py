#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""conv1dpar.py: 
    CNN with parallels convolutional layers.
    For baseline implementation credits to:
        * https://missinglink.ai/guides/keras/keras-conv1d-working-1d-convolutional-neural-networks-keras/
        * https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
"""

__authors__ = ["Jarod Duret", "Jonathan Heno"]
__credits__ = ["Jarod Duret", "Jonathan Heno"]
__version__ = "1.5.0"
__maintainers__ = ["Jarod Duret", "Jonathan Heno"]
__email__ = [
    "jarod.duret@alumni.univ-avignon.fr",
    "jonathan.heno@alumni.univ-avignon.fr",
]
__license__ = "MIT"


import tensorflow as tf

from tensorflow.keras import layers


class Conv1DPar(tf.keras.Model):
    """
    `Conv1DPar` is a modular convolutional class aimed at defining a CNN
    architecture, from a list of parallel convolutional layers.

    Attributes
    ----------
    embedding (tensorflow.keras.layers.Embedding): 
        Defines the embeddings that will be used to train/evaluate the model's
        instance.
    dropout (tensorflow.keras.layers.Dropout):
        Dropout to apply at the end of the fully connected layer. 
    batch_norm (tensorflow.keras.layers.BatchNormalization):
        Type of normalization to apply on input.
        This layer is here used to normalize the data after agregating the 
        convolutional outputs.
    convs (list(tensorflow.keras.layers.Conv1D)):
        The convolutional series to include.
    pooling (tensorflow.keras.layers.GlobalMaxPooling1D):
        Type of pooling to apply to each convolutional layer.
    """
    def __init__(
        self,
        out_dim: int,
        voc_len: int,
        emb_dim: int,
        dropout: float,
        num_filters: int,
        convs: list,
        fc_dim: int
    ):
        super(Conv1DPar, self).__init__()

        self.emdedding = layers.Embedding(voc_len, emb_dim)
        self.dropout = layers.Dropout(dropout)
        self.batch_norm = layers.BatchNormalization()
        
        self.convs = [
            (layers.Conv1D(num_filters, conv['kernel'], strides=conv['strides'], padding="valid", activation="relu"))
            for conv in convs
        ]
        self.pooling = layers.GlobalMaxPooling1D()

        self.dense1 = layers.Dense(fc_dim, activation="relu", name="name")
        self.dense2 = layers.Dense(out_dim, activation="softmax", name="predictions")
    

    def call(self, inputs, training=False):
        x = self.emdedding(inputs)

        conv_outs = [self.pooling(conv(x)) for conv in self.convs]
        x = layers.concatenate(conv_outs, axis=1)
        x = self.batch_norm(x)

        x = self.dense1(x)
        x = self.dropout(x, training=training)

        return self.dense2(x)
