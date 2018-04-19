# thanks to Bharath, https://www.kaggle.com/bharathsh/stanford-q-a-json-to-clean-dataframe, for code adapted for this use
#
import pandas as pd
import numpy as np
import json
import string
from util.text_util import get_word2vec_model
from keras.models import Model, Input
from keras.layers import Conv1D, Dense, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.layers import LeakyReLU
import tensorflow as tf
from warnings import warn



def main():
    # word2vec = get_word2vec_model()


    df = import_stanford_qa_data()


def import_stanford_qa_data():
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
    layer = LeakyReLU()(layer)
    # layer = GlobalAveragePooling1D()(layer)
    # layer = Dense(64, kernel_initializer='truncated_normal', dropout=0.5)(layer)
    # layer = Dense(32, kernel_initializer='truncated_normal', dropout=0.5)(layer)
    # layer = Dense(64, kernel_initializer='truncated_normal', dropout=0.5)(layer)
    #
    # layer = LeakyReLU()(layer)
    model = Model(inputs=inp, outputs=layer)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')
    model.summary()
    return model



def example_map_matrix_to_text(matrix, words):
    print(len(words))
    print(matrix.shape)
    sums = np.sum(matrix, axis=1)
    for n, word in enumerate(words):
        print("{}: {}".format(word, sums[n]))

model = conv_test()

question, context, answer_start, answer_text = extract_fields(df, 1)

mat, words = text_to_matrix(question, word2vec)

mat, words, answer = text_to_matrix_with_answer(
    context, answer_text, answer_start, word2vec)

for w, ans in zip(words, answer):
    print((w, ans))

example_map_matrix_to_text(mat, words)


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







