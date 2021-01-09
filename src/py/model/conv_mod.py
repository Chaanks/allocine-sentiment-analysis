#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""cnn_expanded.py: Modular CNN architecture based on configuration."""

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

import const

from tensorflow.keras import layers, models


class ConvMod(tf.keras.Model):
    def __init__(
        self,
        out_dim: int,
        voc_len: int,
        emb_dim: int,
        layers: list,
    ):
        super(ConvMod, self).__init__()

        self.model = models.Sequential()

        for layer in layer_series:
            if 'dropout' in layer:
                self.model.add(layers.Dropout(layer['dropout']))
            elif 'conv1d' in layer:
                conv = layer['conv1d']
                self.model.add(layers.Conv1D(conv['filters'], conv['kernel'], strides=conv['strides'], padding='valid', activation="relu"))
            elif 'dense' in layer:
                self.model.add(layers.Dense(layer['dense']), activation='relu')
            elif isinstance(layer, str):
                if layer == 'global_max_pooling':
                    self.model.add(layers.GlobalMaxPooling1D())
                elif layer == 'flatten':
                    self.model.add(layers.Flatten())
                else:
                    raise ValueError(const.UNKNOWN_LAYER_TYPE(layer))
        
        self.model.add(layers.Dense(out_dim, activation="softmax", name="predictions"))


    def call(self, inputs, training=False):
        x = self.emdedding(inputs)

        for layer in self.model:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        return x
