import pandas as pd
import numpy as np
import json
import string
from util.text_util import get_word2vec_model
from warnings import warn




def get_word2vec_and_stanford_qa():
    word2vec = get_word2vec_model()
    df = import_stanford_qa_data()
    return word2vec, df



def import_stanford_qa_data():
    # thanks to Bharath, https://www.kaggle.com/bharathsh/stanford-q-a-json-to-clean-dataframe,
    # for code adapted for this use
    #
    data_path = "D:/Datasets/stanford-question-answering-dataset/"
    train_path = data_path + 'train-v1.1.json'
    path = ['data', 'paragraphs', 'qas', 'answers']
    with open(train_path) as fh:
        raw_json = json.loads(fh.read())

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



def example_map_matrix_to_text(matrix, words):
    print(len(words))
    print(matrix.shape)
    sums = np.sum(matrix, axis=0)
    for n, word in enumerate(words):
        print("{}: {}".format(word, sums[n]))



def one_question_test(df, word2vec, model):
    question, context, answer_start, answer_text = extract_fields(df, 1)
    # question_reader, text_reader = get_readers()
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
