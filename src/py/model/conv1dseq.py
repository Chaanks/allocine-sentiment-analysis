#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""conv1dseq.py: 
    CNN defined with a sequence of convolutional layers given a configuration 
    file. For our baseline implementation credits to:
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


import tensorflow

import const


class Conv1DSeq(tensorflow.keras.Model):
    """
    `Conv1DSeq` is a modular convolutional class aimed at defining an
    architecture from a list of `keras.layers` entities stored in a 
    `tensorflow.keras.models.Sequential()` module.

    Attributes
    ----------
    embedding (tensorflow.keras.layers.Embedding): 
        Defines the embeddings that will be used to train/evaluate the model's
        instance.
    model (tensorflow.keras.models.Sequential):
        The sequence of layers to include during the processing phase. Those 
        are defined in the configuration file, in the `model.layers` section.
    """
    def __init__(
        self, out_dim: int, voc_len: int, emb_dim: int, layers: list,
    ):
        super(Conv1DSeq, self).__init__()

        self.embedding = tensorflow.keras.layers.Embedding(voc_len, emb_dim)

        self.model = tensorflow.keras.models.Sequential()

        for layer in layers:
            if "dropout" in layer:
                self.model.add(tensorflow.keras.layers.Dropout(layer["dropout"]))
            elif "conv1d" in layer:
                conv = layer["conv1d"]
                self.model.add(
                    tensorflow.keras.layers.Conv1D(
                        conv["filters"],
                        conv["kernel_size"],
                        strides=conv["strides"],
                        padding="valid",
                        activation="relu",
                    )
                )
            elif "dense" in layer:
                self.model.add(
                    tensorflow.keras.layers.Dense(layer["dense"], activation="relu")
                )
            elif isinstance(layer, str):
                if layer == "global_max_pooling":
                    self.model.add(tensorflow.keras.layers.GlobalMaxPooling1D())
                elif layer == "flatten":
                    self.model.add(tensorflow.keras.layers.Flatten())
                else:
                    raise ValueError(const.UNKNOWN_LAYER_TYPE(layer))

        # Output layer
        self.model.add(
            tensorflow.keras.layers.Dense(
                out_dim, activation="softmax", name="predictions"
            )
        )

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        return self.model(x, training=training)
