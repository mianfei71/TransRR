# -*- coding: utf-8 -*-
"""
Created on 2023-01
@author: ZQ
"""
from keras.layers import *
import tensorflow as tf
import keras
import numpy as np

def positional_embedding(maxlen, model_size):
    PE = np.zeros((maxlen, model_size))
    for i in range(maxlen):
        for j in range(model_size):
            if j % 2 == 0:
                PE[i, j] = np.sin(i / 10000 ** (j / model_size))
            else:
                PE[i, j] = np.cos(i / 10000 ** ((j-1) / model_size))
    PE = tf.constant(PE, dtype=tf.float32)
    return PE

def kernel_inception(x, filter):
    pathway1=Conv1D(filters=filter,kernel_size=1,padding='same',activation='relu')(x)
    pathway2=Conv1D(filters=filter,kernel_size=3,padding='same',activation='relu')(x)
    pathway3=Conv1D(filters=filter,kernel_size=5,padding='same',activation='relu')(x)
    pathway4=Conv1D(filters=filter,kernel_size=7,padding='same',activation='relu')(x)
    out = concatenate([pathway1, pathway2, pathway3, pathway4], axis=-1)
    return out

def dilation_inception(x, filter):
    pathway1 = Conv1D(filters=filter, kernel_size=5, dilation_rate=1, activation="relu", padding="same")(x)
    pathway2 = Conv1D(filters=filter, kernel_size=5, dilation_rate=4, activation="relu", padding="same")(x)
    pathway3 = Conv1D(filters=filter, kernel_size=5, dilation_rate=8, activation="relu", padding="same")(x)
    pathway4 = Conv1D(filters=filter, kernel_size=5, dilation_rate=16, activation="relu", padding="same")(x)
    out = concatenate([pathway1, pathway2, pathway3, pathway4], axis=-1)
    return out

def transformer_encoder(inputs, head_size=256, num_heads=8, dropout=0.2):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = x + inputs
    res = LayerNormalization(axis=1, epsilon=1e-6)(x)

    x = kernel_inception(res, filter=32)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=32, kernel_size=1)(x)

    x = dilation_inception(x, filter=32)
    x = Conv1D(filters=32, kernel_size=5, dilation_rate=16, activation="relu", padding="same")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = x + res
    x = LayerNormalization(axis=1, epsilon=1e-6)(x)

    return x

def TransRR(win_size, num_transformer_blocks=4, mlp_dropout=0.2):
    # Input layer
    input1 = Input(shape=(win_size, 1))
    input2 = Input(shape=(win_size, 1))
    pos_embedding = positional_embedding(win_size, 1)
    pos_wise_input1 = input1 + pos_embedding
    pos_wise_input2 = input2 + pos_embedding
    input_layer = tf.concat([pos_wise_input1, pos_wise_input2], 2)
    x = input_layer

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x)

    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(mlp_dropout)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(mlp_dropout)(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x)

    return keras.Model([input1, input2], outputs)