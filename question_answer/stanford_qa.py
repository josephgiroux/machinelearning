
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


#
# import numpy as np

class ContextRepeat(Layer):

    def __init__(self, **kwargs):
        super(ContextRepeat, self).__init__(**kwargs)

    def call(self, inputs):
        timed_inputs, fixed_inputs = inputs
        n_repeats = tf.shape(timed_inputs)[0:1]
        print("Timed inputs:", timed_inputs.shape)
        print("Fixed inputs:", fixed_inputs.shape)

        print("Tf timed inputs:", tf.shape(timed_inputs))
        print("N repeats", n_repeats, n_repeats.shape)
        n_repeats = tf.convert_to_tensor([1, n_repeats[0]])
        print("N repeats", n_repeats, n_repeats.shape)
        n_fixed_features = tf.shape(fixed_inputs)[-1]
        fixed_inputs = tf.reshape(
            fixed_inputs, shape=(n_fixed_features,))


        n_timesteps = tf.shape(timed_inputs)[-2:-1]
        tiled = tf.tile(fixed_inputs, n_timesteps)
        print("TILED shape: ", tf.shape(tiled), tiled.shape)

        reshape = tf.shape(fixed_inputs)

        print("RESHAPE shape", tf.shape(reshape), reshape.shape)
        print("n_timesteps shape", tf.shape(n_timesteps), n_timesteps.shape)

        print("n_timesteps[0] shape", tf.shape(n_timesteps[0]), n_timesteps[0].shape)
        new_shape = [1, n_timesteps[0], n_fixed_features]

        print(new_shape)
        matrix = tf.reshape(tiled, new_shape)
        combined_matrix = tf.concat((timed_inputs, matrix), axis=-1)
        print("MATRIX", matrix.shape)
        # matrix = tf.reshape(tf.tile(
        #     fixed_inputs, n_repeats
        #     ), [n_repeats] + list(fixed_inputs.shape))
        return combined_matrix


    def compute_output_shape(self, input_shape):

        timed_shape, fixed_shape = input_shape
        n_output_features = timed_shape[-1] + fixed_shape[-1]
        return (
            timed_shape[0],
            timed_shape[1],
            n_output_features)

pred = one_question_test(df, word2vec)
# model = combined_network()



def import_stanford_qa_data():
    # thanks to Bharath, https://www.kaggle.com/bharathsh/stanford-q-a-json-to-clean-dataframe,
    # for code adapted for this use
    #
    data_path = "D:/Datasets/stanford-question-answering-dataset/"
    train_path = data_path + 'train-v1.1.json'
    path = ['data', 'paragraphs', 'qas', 'answers']
    with open(train_path) as fh:
        raw_json = json.loads(fh.read())
    print(raw_json)

    js = pd.io.json.json_normalize(raw_json, path)
    m = pd.io.json.json_normalize(raw_json, path[:-1])
    r = pd.io.json.json_normalize(raw_json, path[:-2])

    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx

    df = pd.concat([
        m[['id', 'question', 'context']].set_index('id'),
        js.set_index('q_idx')], 1).reset_index()

    df['c_id'] = df['context'].factorize()[0]
    df.head()
    return df
#
# one_row = df['id', :]
#
# for row in df:
#     print(row)
#     break
# row = df.iloc[1, :]
# print(row)


def extract_fields(df, idx):
    row = df.iloc[idx]
    question, context, answer_start, answer_text = row.loc[[
        'question', 'context', 'answer_start', 'text']]
    return question, context, answer_start, answer_text


def clean_for_w2v(word):
    rtn = ""
    for ch in word:
        if ch in string.ascii_letters:
            rtn += ch
        elif ch in string.digits:
            rtn += "#"
    return rtn




def get_vec(word, model):
    try:
        return model.get_vector(word)
    except KeyError:
        try:
            return model.get_vector(word.lower())
        except KeyError:
            return None


def text_to_matrix(text, model):
    words = text.split(' ')


    vectors = []
    remaining_words = []
    for word in words:
        vec = get_vec(clean_for_w2v(word), model)
        if vec is None:
            continue
        vectors.append(vec)
        remaining_words.append(word)

    matrix = np.stack(vectors)
    return matrix, remaining_words



def text_to_matrix_with_answer(text, answer, answer_index, model):
    # first remove punctuation
    words = text.split(' ')
    answer_words = answer.split(' ')
    in_answer = False
    num_answer_words = len(answer_words)
    answer_words_found = 0
    vectors = []
    answer_flags = []
    remaining_words = []
    curr_char_idx = 0
    for n, word in enumerate(words):
        if curr_char_idx == answer_index:
            in_answer = True
        if in_answer:
            if not answer_words:
                in_answer = False
                if answer_words_found != num_answer_words:
                    print("Correct answer was not found in string!")
                    print("Answer: {}".format(answer))
                    print("Text: {}".format(text))
                    raise ValueError("Correct answer was not found in string")
            elif word == answer_words[0]:
                answer_words_found += 1
                answer_words = answer_words[1:]
            else:
                print("Wrong answer words")
                print("Answer: {}".format(answer))
                print("Text: {}".format(text))
                raise ValueError("Words came out of order.")


        curr_char_idx += 1 + len(word)
        vec = get_vec(clean_for_w2v(word), model)
        if vec is None:
            continue
        vectors.append(vec)
        remaining_words.append(word)
        answer_flags.append(np.array(
            int(in_answer)))

    assert len(answer_flags) == len(remaining_words)
    matrix = np.stack(vectors)
    answer_flags = np.array(answer_flags)
    if np.sum(answer_flags)==0:
        print("Answer: {}".format(answer))
        print("Text: {}".format(text))
        warn("Answer words not found in word2vec!")
    return matrix, remaining_words, answer_flags

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



def dense_interpreter_network_connected(
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

    def loss_fn(y_true, y_pred, pos_weight=10):
        return tf.nn.weighted_cross_entropy_with_logits(
            targets=y_true,
            logits=y_pred,
            pos_weight=pos_weight,
            name=None)

    # model.summary()
    return layer

interp = dense_interpreter_network()


def example_map_matrix_to_text(matrix, words):
    print(len(words))
    print(matrix.shape)
    sums = np.sum(matrix, axis=0)
    for n, word in enumerate(words):
        print("{}: {}".format(word, sums[n]))



def combined_network(
        n_word_features=300,
        text_conv_specifications=None,
        question_conv_specifications=None,
        question_pooling=GlobalAveragePooling1D,
        ):

    text_conv_specifications = text_conv_specifications or [
        (2, 16),
        (3, 32),
        (4, 64),
        (5, 128),
        (6, 256), # second pair of last element is number of output features per word
    ]

    (
        (question_reader, question_inputs, question_outputs,),
        (text_reader, text_inputs, text_outputs,)
    ) = get_readers()

    dense_out = dense_interpreter_network_connected(
        question_outputs, text_outputs, 300)

    # print(conv.shape)
    # if pooling is not None:
    #     conv = pooling()(conv)
    model = Model(inputs=[question_inputs, text_inputs], outputs=dense_out)

    def loss_fn(y_true, y_pred, pos_weight=10):
        return tf.nn.weighted_cross_entropy_with_logits(
            targets=y_true,
            logits=y_pred,
            pos_weight=pos_weight,
            name=None)
    # model.summary()
    model.compile(optimizer=Adam(lr=0.0001), loss=loss_fn)
    return model

combined_network()



def get_readers():
    question_reader_layers = [
        (2, 4),
        (3, 8),
        (4, 16),
        (5, 32),
        (6, 64),
    ]
    text_reader_layers = [
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



question_reader, text_reader = get_readers()



def one_question_test(df, word2vec):
    question, context, answer_start, answer_text = extract_fields(df, 1)
    # question_reader, text_reader = get_readers()
    model = combined_network()
    question_mat, question_words = text_to_matrix(
        question, word2vec)

    text_mat, text_words, answer = text_to_matrix_with_answer(
        context, answer_text, answer_start, word2vec)

    def add_dim(mat):
        return mat.reshape([1]+list(mat.shape))

    pred = model.predict(x=[
        add_dim(question_mat), add_dim(text_mat)])

    example_map_matrix_to_text(pred, text_words)
    model.summary()
    model.fit([
        add_dim(question_mat),
        add_dim(text_mat)], add_dim(add_dim(answer)))



    return pred





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







