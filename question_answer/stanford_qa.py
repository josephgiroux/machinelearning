# thanks to Bharath, https://www.kaggle.com/bharathsh/stanford-q-a-json-to-clean-dataframe, for code adapted for this use
#
import pandas as pd
import numpy as np
import json
import string
from util.text_util import get_word2vec_model
from keras.models import Model, Input
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.layers import LeakyReLU
import tensorflow as tf

word2vec = get_word2vec_model()



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

df = import_stanford_qa_data()
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


def text_to_matrix(text, model):
    # first remove punctuation
    words = text.split(' ')

    def get_vec(word):
        try:
            return model.get_vector(word)
        except KeyError:
            try:
                return model.get_vector(word.lower())
            except KeyError:
                return None

    vectors = []
    for word in words:
        vec = get_vec(clean_for_w2v(word))
        if vec is None:
            continue
        vectors.append(vec)

    matrix = np.stack(vectors)
    return matrix



def conv_test(n_inputs=300, lr=0.0001):
    tf.global_variables_initializer()
    inp = Input((None, n_inputs), dtype='float32')
    print(inp.shape)
    layer = inp
    layer = Conv1D(
        32, kernel_size=(4,), padding='same',
        kernel_initializer='glorot_normal')(layer)
    print(layer.shape)
    conv = layer
    layer = LeakyReLU()(layer)

    layer = MaxPooling1D(4)(layer)
    print(layer.shape)
    model = Model(inputs=inp, outputs=conv)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')
    return model



model = conv_test()
print(mat.shape)
print(mat[:,np.newaxis].shape)
print(model.predict(mat.reshape([1]+list(mat.shape))))
out = model.predict(mat.reshape([1]+list(mat.shape)))
print(out.shape)


question, context, answer_start, answer_text = extract_fields(df, 1)

print(question)
print(answer_text)
print(context)

mat = text_to_matrix(question, model)






