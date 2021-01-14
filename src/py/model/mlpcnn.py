import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def create_mlp(preprocessing, inputs):
	# return our model
    body = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="linear")
    ])

    preprocessed_inputs = preprocessing(inputs)
    result = body(preprocessed_inputs)
    
    return tf.keras.Model(inputs, result)

def create_cnn(max_features, embedding_dim, vectorize_layer):
    # A integer input for vocab indices.
    inputs = tf.keras.Input(shape=(None,), dtype="int32")
    

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(max_features, embedding_dim)(inputs)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(128, 3, padding="valid", activation="relu", strides=1)(x)
    x = layers.Conv1D(128, 4, padding="valid", activation="relu", strides=2)(x)
    x = layers.MaxPooling1D()(x)
    x = tf.keras.layers.Flatten()(x)

    model = tf.keras.Model(inputs, x)
    
    # A string input
    inputs = tf.keras.Input(shape=(1,), dtype="string")
    # Turn strings into vocab indices
    indices = vectorize_layer(inputs)
    # Turn vocab indices into predictions
    outputs = model(indices)
    
    return tf.keras.Model(inputs, outputs)

def create_model(feats_encoder, inputs, reviews_encoder, max_features, embedding_dim):
    # create the MLP and CNN models
    mlp = create_mlp(feats_encoder, inputs)
    cnn = create_cnn(max_features, embedding_dim, reviews_encoder)

    # create the input to our final set of layers as the *output* of both
    # the MLP and CNN
    combinedInput = layers.concatenate([mlp.output, cnn.output])

    # our final FC layer head will have two dense layers, the final one
    # being our regression head
    x = layers.Dense(128, activation="relu")(combinedInput)
    x = layers.Dense(10, activation="softmax")(x)
    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input, outputting a single value (the
    # predicted price of the house)
    return tf.keras.Model(inputs=[mlp.input, cnn.input], outputs=x)