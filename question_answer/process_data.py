import pandas as pd
import numpy as np
import json
import string
from util.text_util import get_word2vec_model
from warnings import warn
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
unknown_vectors = defaultdict(lambda: np.random.normal(size=(300,)))
import regex as re

def replace_digits(word):
    replace_digits = re.compile("([0-9]{2,})")
    new_str = str(word)
    matches = replace_digits.finditer(word)
    for m in matches:
        new_str = new_str[:m.start(0)] + "#" * len(m[0]) + new_str[m.end(0):]
    return new_str
"#" * len(m[0])

line = re.sub(
           r"(?i)^.*interfaceOpDataFile.*$",
           "interfaceOpDataFile %s" % fileIn,
           line
       )


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
        elif ch in string.punctuation:
            rtn += '_'

    return rtn.strip('_')




def get_vec(word, model, possible_next=None):
    def _get_vec(word):
        try:
            return model.get_vector(word)
        except KeyError:
            low = word.lower()
            try:
                return model.get_vector(low)
            except KeyError:
                if low in stopwords:
                    return None
                else:
                    print(word)
                    return unknown_vectors[word]

    if False and possible_next:
        # disable this look-ahead for now, a little more complexity than is needed
        possible_combined = _get_vec("_".join([word, possible_next]))
        if possible_combined is not None:
            return possible_combined, 2
    else:
        return _get_vec(word), 1


def text_to_matrix(text, model):
    words = text.split(' ')


    vectors = []
    remaining_words = []
    idx, end = 0, len(words)
    clean_words = list(map(clean_for_w2v, words))
    while idx < end:
        word = clean_words[idx]
        try:
            possible_next = clean_words[idx+1]
        except IndexError:
            possible_next = None

        vec, n = get_vec(word, model, possible_next)
        idx += n
        if vec is None:
            continue
        vectors.append(vec)
        remaining_words.append(word)

    matrix = np.stack(vectors)
    return matrix, remaining_words



def text_to_matrix_with_answer(text, answer, answer_idx, model):
    # first remove punctuation
    words = text.split(' ')
    # clean_words = map(clean_for_w2v, words)\
    answer_words = answer.split(' ')
    # clean_answer_words = map(clean_for_w2v, answer_words)
    in_answer = False
    num_answer_words = len(answer_words)
    answer_words_found = 0
    vectors = []
    answer_flags = []
    remaining_words = []
    curr_char_idx = 0

    for n, word in enumerate(words):
        if curr_char_idx <= answer_idx < curr_char_idx + len(word):
            if answer_idx != curr_char_idx:
                print("Watch out, was {} off because of {}".format(
                    np.abs(answer_idx-curr_char_idx), word[:np.abs(answer_idx-curr_char_idx)]))
            in_answer = True
            # elif (curr_char_idx > answer_idx and curr_char_idx < answer_idx)
            # if we do a look-ahead, here is where we would check if we overlapped
            # the answer with the look-ahead... this is a to-do for later to see if
            # the model needs extra juice
        if in_answer:
            if not answer_words:
                in_answer = False
                if answer_words_found != num_answer_words:
                    print("Correct answer was not found in string!")
                    print("Answer: {}".format(answer))
                    print("Text: {}".format(text))
                    raise ValueError("Correct answer was not found in string")
            elif answer_words[0] in word:
                answer_words_found += 1
                answer_words = answer_words[1:]
            else:
                print("The word was {} and I was expecting {}".format(word, answer_words[0]))
                print("Wrong answer words")
                print("Answer: {}".format(answer))
                print("Text: {}".format(text))
                raise ValueError("Words came out of order.")


        curr_char_idx += 1 + len(word)
        vec, n = get_vec(clean_for_w2v(word), model)
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
        print(answer_words)
        print(words)
        raise ValueError("Answer words not found in word2vec!")
    return (
        matrix,
        remaining_words,
        answer_flags,
        answer_words_found,
        len(remaining_words) - answer_words_found)



def example_map_matrix_to_text(matrix, words):
    sums = np.sum(matrix, axis=0)
    for n, word in enumerate(words):
        print("{}: {}".format(word, sums[n]))



def one_question_test(df, word2vec, model):
    question, context, answer_start, answer_text = extract_fields(df, 1)
    # question_reader, text_reader = get_readers()
    question_mat, question_words = text_to_matrix(
        question, word2vec)

    text_mat, text_words, answer, num_positive, num_negative = text_to_matrix_with_answer(
        text=context, answer=answer_text,
        answer_idx=answer_start, model=word2vec)
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


