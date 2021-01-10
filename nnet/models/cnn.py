import tensorflow as tf
from tensorflow.keras import layers


class CovNet1D(tf.keras.Model):

  def __init__(self, cfg):
    super(CovNet1D, self).__init__()

    self.cfg = cfg
    self.emdedding = layers.Embedding(
      self.cfg['max_features'],
      self.cfg['embedding_dim'],
      embeddings_initializer=tf.keras.initializers.Constant(self.cfg['embedding_matrix']),
      trainable=False
    )
    self.dropout = layers.Dropout(0.5)

    # Conv1D + global max pooling
    self.conv1 = layers.Conv1D(128, 3, padding="valid", activation="relu", strides=1)
    self.conv2 = layers.Conv1D(128, 3, padding="valid", activation="relu", strides=1)
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


class HomeMade(tf.keras.Model):

  def __init__(self, cfg):
    super(HomeMade, self).__init__()

    self.cfg = cfg
    self.emdedding = layers.Embedding(self.cfg['max_features'], self.cfg['embedding_dim'])
    self.dropout = layers.Dropout(0.3)
    self.bn = tf.keras.layers.BatchNormalization()

    self.conv1 = layers.Conv1D(512, 3, activation="relu")
    self.pooling1 = layers.GlobalMaxPooling1D()
    self.conv2 = layers.Conv1D(512, 2, activation="relu")
    self.pooling2 = layers.GlobalMaxPooling1D()

    self.dense2 = layers.Dense(10, activation="softmax", name="cnn")

  def call(self, inputs, training=False):
    x = self.emdedding(inputs)
    x1 = self.conv1(x)
    x1 = self.pooling1(x)

    x2 = self.conv1(x)
    x2 = self.pooling1(x)

    x = tf.keras.layers.concatenate([x1, x2], axis=1)
    x = self.bn(x)
    x = self.dropout(x, training=training)

    return self.dense2(x)