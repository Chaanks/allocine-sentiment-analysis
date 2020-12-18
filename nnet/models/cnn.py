import tensorflow as tf
from tensorflow.keras import layers


class CovNet1D(tf.keras.Model):

  def __init__(self, cfg):
    super(CovNet1D, self).__init__()

    self.cfg = cfg
    self.emdedding = layers.Embedding(self.cfg['max_features'], self.cfg['embedding_dim'])
    self.dropout = layers.Dropout(0.5)

    # Conv1D + global max pooling
    self.conv1 = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)
    self.conv2 = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)
    self.pooling = layers.GlobalMaxPooling1D()

    # We add a vanilla hidden layer:
    self.dense1 = layers.Dense(128, activation="relu")
    self.dense2 = layers.Dense(10, activation="softmax", name="predictions")

  def call(self, inputs, training=False):
    x = self.emdedding(inputs)
    x = self.dropout(x, training=training)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.pooling(x)
    x = self.dense1(x)
    x = self.dropout(x, training=training)
    return self.dense2(x)