import tensorflow as tf
from tensorflow.keras import layers


class CovNet1D(tf.keras.Model):
    def __init__(self, out_dim: int, voc_len: int, emb_dim: int, frame_len:int, dropout: float, strides: int):
        super(CovNet1D, self).__init__()

        self.emdedding = layers.Embedding(voc_len, emb_dim)
        self.dropout = layers.Dropout(dropout)

        # Conv1D + global max pooling
        self.conv1 = layers.Conv1D(
            emb_dim, frame_len, padding="valid", activation="relu", strides=strides
        )
        self.conv2 = layers.Conv1D(
            emb_dim, frame_len, padding="valid", activation="relu", strides=strides
        )
        self.pooling = layers.GlobalMaxPooling1D()

        # We add a vanilla hidden layer:
        self.dense1 = layers.Dense(emb_dim, activation="relu")
        self.dense2 = layers.Dense(out_dim, activation="softmax", name="predictions")

        self.cfg = {"out_dim": out_dim, "voc_len": voc_len, "emb_dim": emb_dim, "frame_len": frame_len, "dropout": dropout, "strides": strides}

    def call(self, inputs, training=False):
        x = self.emdedding(inputs)
        x = self.dropout(x, training=training)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)

        return self.dense2(x)
