import pandas as pd
import numpy as np
import json
import string
from util.text_util import get_word2vec_model
from keras.models import Model, Input
from keras.layers import Conv1D, Dense, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, Dropout, Lambda, Concatenate, \
    TimeDistributed, RepeatVector
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from warnings import warn
from question_answer.layer import ContextRepeat


def conv_reader_network(
        n_features=300,
        lr=0.0001,
        conv_specifications=None,
        pooling=None):
    inp = Input((None, n_features), dtype='float32')

    conv_specifications = conv_specifications or [
        (2, 16),
        (3, 32),
        (4, 64),
        (5, 128),
        (6, 256), # second pair of last element is number of output features per word
    ]
    layer = inp
    for n, (kernel_size, filters) in enumerate(conv_specifications):
        if n:
            combined = Concatenate()([layer, conv])
            layer = combined
        conv = Conv1D(
            filters, kernel_size=(kernel_size,), padding='same',
            kernel_initializer='glorot_normal')(layer)
        conv = LeakyReLU()(conv)

    print(conv.shape)
    if pooling is not None:
        conv = pooling()(conv)
    model = Model(inputs=inp, outputs=conv)

    model.summary()
    return model, inp, conv


def dense_interpreter_network(
        question_inputs,
        text_inputs,
        n_question_features=64,
        n_text_features=256,
        dropout=0.5,
        lr=0.0001):

    neuron_counts = [1024, 256, 64, 16, 1]
    layer = ContextRepeat()([text_inputs, question_inputs])
    layer = TimeDistributed(Dense(neuron_counts[0]))(layer)
    for num_units in neuron_counts[1:]:
        layer = Dense(num_units)(layer)
        layer = Dropout(dropout)(layer)

    return layer



def combined_network(
        n_word_features=300,
        text_reader_layers=None,
        question_reader_layers=None,
        question_pooling=GlobalAveragePooling1D,
        ):

    (
        (question_reader, question_inputs, question_outputs,),
        (text_reader, text_inputs, text_outputs,)
    ) = get_readers(
        text_reader_layers=text_reader_layers,
        question_reader_layers=question_reader_layers)

    dense_out = dense_interpreter_network(
        question_inputs=question_outputs,
        text_inputs=text_outputs)

    model = Model(
        inputs=[question_inputs, text_inputs],
        outputs=dense_out)

    def loss_fn(y_true, y_pred, pos_weight=10):
        return tf.nn.weighted_cross_entropy_with_logits(
            targets=y_true,
            logits=y_pred,
            pos_weight=pos_weight,
            name=None)
    # model.summary()
    model.compile(optimizer=Adam(lr=0.0001), loss=loss_fn)
    return model


def get_readers(
        text_reader_layers=None,
        question_reader_layers=None):

    question_reader_layers = question_reader_layers or [
        (2, 4),
        (3, 8),
        (4, 16),
        (5, 32),
        (6, 64),
    ]
    text_reader_layers = text_reader_layers or [
        (2, 16),
        (3, 32),
        (4, 64),
        (5, 128),
        (6, 256), # second pair of last element is number of output features per word
    ]
    question_reader, question_inputs, question_outputs = conv_reader_network(
        conv_specifications=question_reader_layers,
        pooling=GlobalAveragePooling1D)
    text_reader, text_inputs, text_outputs = conv_reader_network(
        conv_specifications=text_reader_layers)

    return (
        (question_reader, question_inputs, question_outputs,),
        (text_reader, text_inputs, text_outputs,),)


