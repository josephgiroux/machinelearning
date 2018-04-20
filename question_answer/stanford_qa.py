
import pandas as pd
import numpy as np
import json
import string
from util.text_util import get_word2vec_model
from keras.models import Model, Input
from keras.layers import Conv1D, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout, Lambda, Concatenate
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras import backend as K
import tensorflow as tf
from warnings import warn



def main():
    word2vec = get_word2vec_model()
    df = import_stanford_qa_data()
    return word2vec, df
word2vec, df = main()

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
        answer_flags.append(int(in_answer))

    assert len(answer_flags) == len(remaining_words)
    matrix = np.stack(vectors)
    if np.sum(answer_flags)==0:
        print("Answer: {}".format(answer))
        print("Text: {}".format(text))
        warn("Answer words not found in word2vec!")
    return matrix, remaining_words, np.array(answer_flags)

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

    def loss_fn(y_true, y_pred, pos_weight=2):
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
    return model


def dense_interpreter_network(
        n_question_features=64,
        n_text_features=256,
        dropout=0.5,
        lr=0.0001):

    neuron_counts = [1024, 256, 64, 16, 2]
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
    model.compile(optimizer=Adam(lr=lr), loss=loss_fn)
    model.summary()
    return model



interp = dense_interpreter_network()


def example_map_matrix_to_text(matrix, words):
    print(len(words))
    print(matrix.shape)
    sums = np.sum(matrix, axis=1)
    for n, word in enumerate(words):
        print("{}: {}".format(word, sums[n]))





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
    question_reader = conv_reader_network(
        conv_specifications=question_reader_layers,
        pooling=GlobalAveragePooling1D)
    text_reader = conv_reader_network(
        conv_specifications=text_reader_layers)

    return question_reader, text_reader



question_reader, text_reader = get_readers()



def one_question_test(df, word2vec):

    question_reader, text_reader = get_readers()
    question_mat, question_words = text_to_matrix(
        question, word2vec)

    text_mat, text_words, answer = text_to_matrix_with_answer(
        context, answer_text, answer_start, word2vec)

    def add_dim(mat):
        return mat.reshape([1]+list(mat.shape))

    question_pred = question_reader.predict(add_dim(question_mat))
    text_pred = text_reader.predict(add_dim(text_mat))

    interp = dense_interpreter_network()

    text_word = text_pred[0][0]

    interp_input = np.concatenate((question_pred[0], text_pred[0][0]))
    interp_pred = interp.predict(add_dim(interp_input)


    return question_pred, text_pred


question_pred, text_pred = one_question_test(df, word2vec)

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







