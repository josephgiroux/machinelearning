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
import unidecode
from question_answer.util import pickle_me, unpickle_me

re.DEFAULT_VERSION = re.VERSION1
replace_digits_rgx = re.compile("([0-9,]+[0-9]+)")
o_point_rgx = re.compile("[.][0-9]+[%]")
splitter_rgx = re.compile("[— .'\"&/\-–()~]")

suffix_replacements = {
    'ised': 'ized',
    'sation': 'zation',
    'ysed': 'yzed',
    'aton': 'ation',
    'oured': 'ored',
    'ourite': 'orite',
}

VECTOR_PICKLE = "C:/Users/Joseph Giroux/Datasets/vector.pkl"
MODEL_PATH = "C:/Users/Joseph Giroux/Datasets/qa_model.h5"

VECTOR_BATCH_PICKLE = "C:/Users/Joseph Giroux/Datasets/vector_{}.pkl"
DF_PICKLE = "C:/Users/Joseph Giroux/Datasets/df.pkl"

def replace_digits(word):
    new_str = str(word)
    matches = replace_digits_rgx.finditer(word)
    for m in matches:
        new_str = new_str[:m.start(0)] + "#" * len(m[0]) + new_str[m.end(0):]
    return new_str



def get_word2vec_and_stanford_qa_from_scratch():
    word2vec = get_word2vec_model()
    df = import_stanford_qa_data()
    vectors = get_vector_information(df, word2vec)
    return word2vec, df, vectors


def save_vectors_and_df(vectors, df):
    pickle_me(vectors, VECTOR_PICKLE)
    pickle_me(df, DF_PICKLE)

def save_model(model):
    model.save(MODEL_PATH)


def get_stanford_qa_and_vectors_pickled():
    df = unpickle_me(DF_PICKLE)
    vectors = unpickle_me(VECTOR_PICKLE)
    return df, vectors

def pickle_vectors_in_batches(
        vectors, batch_size=10000):

    for n in range(1, int(len(vectors)/batch_size)+2):
        start = 0 + batch_size * (n-1)
        end = start + 10000
        batch = vectors[start:end]
        pickle_file = VECTOR_BATCH_PICKLE.format(n)
        pickle_me(batch, pickle_file)

def pad_vectors(vector_batch):
    longest_question_vector = 0
    longest_text_vector = 0


def split_with_dollarsign(text):
    words = splitter_rgx.split(text)
    rtn = []
    for word in words:
        # whole lot hassle to split off dollar sign as its own word
        # while not losing track of the answer index

        subparsed = subparse_for_dollarsigns(word)
        rtn += subparsed
    return rtn

def subparse_for_dollarsigns(word):
    rtn = []
    for ch in ['CAD$', 'US$', 'CN¥', 'GB£', 'C$', 'NZ$', 'NT$', 'A$', 'S$', 'R$', '#$', '$', '£', '¥', '€']:
        if ch in word:
            if word == ch:
                return [word, '']
            for n, w in enumerate(word.split(ch)):
                if n:
                    rtn.append(ch)
                if w or n:
                    rtn.append(w)
            return rtn
    else:
        return [word]


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
    df = one_off_corrections(df)
    return df


def one_off_corrections(df):
    for idx in range(df.shape[0]):
        answer = df.loc[idx, 'text']
        if answer.startswith(". "):
            df.loc[idx, 'text'] = answer[2:]
            df.loc[idx, 'answer_start'] += 2
    df.loc[55764, 'answer_start'] = 370
    df.loc[55764, 'text'] = 'lower-elevation areas of the Piedmont'
    df.loc[70455, 'answer_start'] = 495
    df.loc[70455, 'text'] = 'somehow-belligerent'
    df.loc[85016, 'answer_start'] = 387

    df.loc[85016, 'text'] = '38%'
    # df.loc[36157, 'answer_start'] = 179
    # df.loc[36157, 'text'] = '6-months'
    #
    # df.loc[36212, 'answer_start'] = 132
    # df.loc[36212, 'text'] = 'as white or "other."'
    #
    # df.loc[36257, 'answer_start'] = 371
    # df.loc[36257, 'text'] = 'Diverse immigration'
    return df



def extract_fields(df, idx):
    row = df.iloc[idx]
    question, context, answer_start, answer_text = row.loc[[
        'question', 'context', 'answer_start', 'text']]

    return question, context, answer_start, answer_text


def clean_for_w2v(word):
    rtn = ""
    word = replace_digits(word)
    for ch in word:
        if ch.isalnum():
            rtn += ch
        elif ch in '$£¥€#':
            rtn += ch
        elif ch in string.punctuation:
            # Possibly save/split out meaningful punctuation, like $ --
            # a little too much trouble now
            # print(ch)
            # rtn += '_'
            pass

    return rtn.strip('_')


def sep_digits(word):
    first = ""
    second = ""
    dig_first = None
    for n, ch in enumerate(word):
        if not n:
            dig_first = ch=='#'

        if ch == '#':
            if dig_first:
                first += ch
            else:
                second += ch
        else:
            if dig_first:
                second += ch
            else:
                first += ch

    return first, second

def get_vec(word, model, possible_next=None):
    if not word:
        return [None], [None], 1
    if word in ['US$', 'CAD$', 'C$', 'NZ$', 'NT$', 'A$', 'S$', 'R$']:
        return [model.get_vector('$')], [word], 0
    elif word == 'CN¥':
        return [model.get_vector('¥')], [word], 0
    elif word == 'GB£':
        return [model.get_vector('£')], [word], 0
    elif word in ['£', '$', '¥', '€']:
        return [model.get_vector(word)], [word], 0
    elif word == '#$':
        return [None], [word], 0

    orig = word
    def _get_vec(word):
        try:
            return [model.get_vector(word)], [word], 1
        except KeyError:
            word = word.lower()
            try:
                return [model.get_vector(word)], [word], 1
            except KeyError:
                if word in stopwords:
                    return [None], [word], 1
                else:

                    for k,v in suffix_replacements.items():
                        if word.endswith(k):
                            word = word[:-len(k)] + v

                    word = unidecode.unidecode(word)
                    try:
                        return [model.get_vector(word)], [word], 1
                    except KeyError:
                        print(orig, word)

                        if '#' in word:
                            try:
                                first, second = sep_digits(word)
                                rtn = []
                                for part in [first, second]:
                                    if part:
                                        try:
                                            rtn.append(model.get_vector(part))
                                        except KeyError:
                                            rtn.append(unknown_vectors[part])
                                return rtn, [first, second], 1
                            except KeyError:
                                return [unknown_vectors[word]], [word], 1
                        else:
                            return [unknown_vectors[word]], [word], 1

    if False and possible_next:
        # disable this look-ahead for now, a little more complexity than is needed
        possible_combined = _get_vec("_".join([word, possible_next]))
        if possible_combined is not None:
            return possible_combined, 2
    else:
        # hack here -- since $ has to be split by
        # itself, in order to keep the character counter
        # from advancing an extra character every time a $
        # is seen, it has to return 0 in that case.
        return _get_vec(word)


def text_to_matrix(text, model):
    words = split_with_dollarsign(text)


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

        vecs, words, n = get_vec(word, model, possible_next)
        idx += n + len(word)

        for vec, word in zip(vecs, words):
            if vec is None:
                continue
            vectors.append(vec)
            remaining_words.append(word)

    matrix = np.stack(vectors)
    return matrix, remaining_words





def text_to_matrix_with_answer(text, answer, answer_idx, model):
    # first remove punctuation
    words = split_with_dollarsign(text)
    # clean_words = map(clea_rn_for_w2v, words)\
    answer_words = split_with_dollarsign(answer)
    # clean_answer_words = map(clean_for_w2v, answer_words)
    in_answer = False
    num_answer_words = len(answer_words)
    answer_words_found = 0
    vectors = []
    answer_flags = []
    remaining_words = []
    curr_char_idx = 0

    for n, word in enumerate(words):

        # print((word, curr_char_idx, answer_idx))
        if curr_char_idx == answer_idx or (
                word and curr_char_idx <= answer_idx < (curr_char_idx + (len(word) or 1))):
            print((word, curr_char_idx, answer_idx))
            if answer_idx != curr_char_idx:
                print("Watch out, was {} off because of {}".format(
                    np.abs(answer_idx-curr_char_idx),
                    word[:np.abs(answer_idx-curr_char_idx)]))
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
                print(words)
                print(answer_words)
                print("Wrong answer words")
                print("Answer: {}".format(answer))
                print("Text: {}".format(text))
                raise ValueError("Words came out of order.")


        vecs, words, inc = get_vec(clean_for_w2v(word), model)
        curr_char_idx += inc + len(word)

        for vec, word in zip(vecs, words):
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
        print(answer_words_found)
        print(answer_flags)
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



def get_vector_information(df, word2vec, n=10):
    vectors = []

    df['text_matrix'] = None
    df['question_matrix'] = None
    df['answer_vector'] = None
    n_rows = df.shape[0]
    print(df.shape[0])
    total_pos = 0
    total_neg = 0
    # for idx in range(n_rows):
    #     print(idx)
    for idx in range(n_rows): #  range(n_rows):
        print(idx)
        question, context, answer_start, answer_text = extract_fields(df, idx)
        print(question)
        print(answer_start)
        print(answer_text)
        # especially annoying special cases
        if answer_text != '.50-inch' and o_point_rgx.match(answer_text):
            answer_text = "0" + answer_text
            answer_start -= 1

        question_mat, question_words = text_to_matrix(
            text=question, model=word2vec)
        text_mat, text_words, answer_vec, num_pos, num_neg = text_to_matrix_with_answer(
            text=context, answer=answer_text,
            answer_idx=answer_start, model=word2vec)

        # df.loc[idx, 'text_matrix'] = text_mat
        # df.loc[idx,'question_matrix'] = question_mat
        # df.loc[idx, 'answer_vector'] = answer_vec

        df.iloc[idx].at['text_matrix'] = text_mat
        df.iloc[idx].at['question_matrix'] = question_mat
        df.iloc[idx].at['answer_vector'] = answer_vec

        row = dict(
            idx=idx,
            question=question,
            text=context,
            answer=answer_text,
            text_matrix=text_mat,
            question_matrix=question_mat,
            answer_vector=answer_vec,)
        vectors.append(row)
        total_pos += num_pos
        total_neg += num_neg

    print(
        "On average there were {} answer words and {} other words per row."
        "\nPositive should be weighted about {}.".format(
            total_pos / n_rows, total_neg / n_rows, total_neg / total_pos))
    return vectors



def get_train_test_valid_groups(vectors):
    total = 87599
    assert total == len(vectors)
    train = 1
    test = 2
    valid = 3
    final_valid = 4
    assignments = []


    train_text_x = []
    train_question_x = []

    train_y = []
    test_text_x = []
    test_question_x = []
    test_y = []
    valid_text_x = []
    valid_question_x = []

    valid_y = []

    final_valid_text_x = []
    final_valid_question_x = []
    final_valid_y = []

    def assign(idx):
        if idx >= 80000:
            return final_valid
        else:
            grp = idx % 8
            if grp == 7:
                return valid
            if grp == 6:
                return test
            else:
                return train

    def add_dim(mat):
        return mat.reshape([1] + list(mat.shape))

    for n in range(total):
        group = assign(n)
        if group == train:
            train_text_x.append(add_dim(vectors[n]["text_matrix"]))
            train_question_x.append(add_dim(vectors[n]["question_matrix"]))
            train_y.append(add_dim(add_dim(vectors[n]["answer_vector"])))
        elif group == test:
            test_text_x.append(add_dim(vectors[n]["text_matrix"]))
            test_question_x.append(add_dim(vectors[n]["question_matrix"]))
            test_y.append(add_dim(add_dim(vectors[n]["answer_vector"])))
        elif group == valid:
            valid_text_x.append(add_dim(vectors[n]["text_matrix"]))
            valid_question_x.append(add_dim(vectors[n]["question_matrix"]))
            valid_y.append(add_dim(add_dim(vectors[n]["answer_vector"])))
        elif group == final_valid:
            final_valid_text_x.append(add_dim(vectors[n]["text_matrix"]))
            final_valid_question_x.append(add_dim(vectors[n]["question_matrix"]))
            final_valid_y.append(add_dim(add_dim(vectors[n]["answer_vector"])))

    all_train = (train_text_x, train_question_x, train_y)
    all_test = (test_text_x, test_question_x, test_y)
    all_valid = (valid_text_x, valid_question_x, valid_y)
    all_final_valid = (final_valid_text_x, final_valid_question_x, final_valid_y)
    return (all_train, all_test, all_valid, all_final_valid)


def consolidate_text_vectors(df, raw_vectors):
    text_vectors = dict()
    for idx, row in enumerate(raw_vectors):
        c_id = df.iloc[idx].loc['c_id']
        text_vectors[c_id] = row["text_matrix"]

    return text_vectors


def remove_text_vectors(raw_vectors):
    for row in raw_vectors:
        row.pop('text_matrix', None)


def pad_text_vectors(text_vectors, n_features=300):
    longest_text = len(max(text_vectors.items(), key=len))
    text_vectors = {k: pad_vector_list(
        v, desired_len=longest_text, n_features=n_features
        ) for k,v in text_vectors.items()}
    return text_vectors


def pad_vector_list(vectors, desired_len, n_features=300):
    diff = desired_len - len(vectors)
    if diff < 0:
        raise ValueError("Can't pad to a shorter length")

    if diff:
        post_pad = diff // 2
        pre_pad = diff - post_pad
        vectors = ([np.zeros(n_features, )] * pre_pad) + vectors + (
                [np.zeros(n_features, )] * post_pad)

    return vectors


def pad_question_vectors(vectors, n_features=300):
    longest_question = 0
    for row in vectors:
        longest_question = max(
            longest_question,
            len(row["question_matrix"]))

    for row in vectors:
        row["question_matrix"] = pad_vector_list(
            row["question_matrix"],
            desired_len=longest_question,
            n_features=n_features)

    return vectors

