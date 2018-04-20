
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



def main():
    word2vec = get_word2vec_model()
    df = import_stanford_qa_data()
    return word2vec, df
word2vec, df = main()


def conv_test(n_features=300, lr=0.0001):
    tf.global_variables_initializer()
    inp = Input((None, n_features), dtype='float32')
    layer = inp
    layer = Conv1D(
        32, kernel_size=(6,), padding='same',
        kernel_initializer='glorot_normal')(layer)
    # activations = layer
    # layer =
    # layer = LeakyReLU()(layer)
    layer = Lambda(
        lambda x: K.sum(x, axis=0),
        output_shape=lambda s: (s[1], s[2]))(layer)
    # layer = GlobalAveragePooling1D()(layer)
    # layer = Dense(64, kernel_initializer='truncated_normal', dropout=0.5)(layer)
    # layer = Dense(32, kernel_initializer='truncated_normal', dropout=0.5)(layer)
    # layer = Dense(64, kernel_initializer='truncated_normal', dropout=0.5)(layer)
    #
    # layer = LeakyReLU()(layer)

    def loss_fn(y_true, y_pred, pos_weight=10):
        return tf.nn.weighted_cross_entropy_with_logits(
            targets=y_true,
            logits=y_pred,
            pos_weight=pos_weight,
            name=None)

    model = Model(inputs=inp, outputs=layer)
    model.compile(optimizer=Adam(lr=lr), loss=loss_fn)
    model.summary()
    return model






def dense_interpreter_network(
        n_question_features=64,
        n_text_features=256,
        dropout=0.5,
        lr=0.0001):

    neuron_counts = [1024, 256, 64, 16, 1]
    inp = Input((n_question_features + n_text_features,))
    layer = inp
    for n, num_units in enumerate(neuron_counts):
        if n:
            layer = LeakyReLU()(layer)
        layer = Dense(num_units)(layer)
        layer = Dropout(dropout)(layer)

    def loss_fn(y_true, y_pred, pos_weight=2):
        return tf.nn.weighted_cross_entropy_with_logits(
            targets=y_true,
            logits=y_pred,
            pos_weight=pos_weight,
            name=None)

    model = Model(inputs=inp, outputs=layer)
    # model.summary()
    return model, inp, layer



question_reader, text_reader = get_readers()






pred = one_question_test(df, word2vec)

model.fit(x, answer)

print(question_pred.shape)
print(text_pred.shape)


# read a question and text






# model = conv_test()

question, context, answer_start, answer_text = extract_fields(df, 1)

mat, words = text_to_matrix(question, word2vec)

mat, words, answer = text_to_matrix_with_answer(
    context, answer_text, answer_start, word2vec)

x = mat.reshape([1]+list(mat.shape))

pred = reader.predict(x)
pred = model.predict(x)
print(pred.shape)
model.fit(x, answer)

print(pred)
for n, (w, ans) in enumerate(zip(words, answer)):
    print((w, ans, pred[0][n].sum()))

example_map_matrix_to_text(mat, words)

example_map_matrix_to_text(pred[0], words)

print(mat.shape)

pred = model.predict(mat.reshape([1]+list(mat.shape)))
print(pred)
print(pred.shape)


print(mat[:,np.newaxis].shape)
print(model.predict(mat.reshape([1]+list(mat.shape))))
out = model.predict(mat.reshape([1]+list(mat.shape)))
print(out.shape)



print(question)
print(answer_text)
print(context)







