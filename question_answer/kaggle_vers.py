# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import regex as re
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import pickle
import json
import string
import os


# won't be able to load all of word2vec nor save the large amounts of data
# on kaggle kernel, so set a demo mode that will return small zero vectors
# for testing purposes.
KAGGLE_DEMO = True

# set these paths according to your system

STANFORD_DATA_PATH = "D:/Datasets/stanford-question-answering-dataset/"
WORD2VEC_PATH = '/path/to/word2vec/GoogleNews-vectors-negative300.bin'
OUTPUT_DATA_PATH = "D:/Datasets/kaggle_out/"

##############################################################################

re.DEFAULT_VERSION = re.VERSION1
replace_digits_rgx = re.compile(r"([0-9,]+[0-9]+)")
o_point_rgx = re.compile(r"[.][0-9]+[%]")
splitter_rgx = re.compile(r"[ء:— .'\"&/\-–()~]")

suffix_replacements = {
    'ised': 'ized',
    'sation': 'zation',
    'ysed': 'yzed',
    'aton': 'ation',
    'oured': 'ored',
    'ourite': 'orite',
}

DF_PICKLE = os.path.join(OUTPUT_DATA_PATH, "qa_dataframe.pkl")
TEXT_VECTOR_PICKLE = os.path.join(OUTPUT_DATA_PATH, "qa_text_vector.pkl")
TEXT_WORDS_PICKLE = os.path.join(OUTPUT_DATA_PATH, "qa_text_words.pkl")
QUESTION_DATA_PICKLE = os.path.join(OUTPUT_DATA_PATH, "qa_data_vector.pkl")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
def main():
    print("Getting all data from scratch.")
    word2vec, df, text_vectors, text_words, question_data = get_word2vec_and_stanford_qa_from_scratch()
    print("Saving results.")
    save_vectors_and_df(question_data=question_data, text_vectors=text_vectors, text_words=text_words, df=df)
    print("Saving successful.")
    print("Loading results.")
    df, question_vectors, text_vectors, text_words = get_stanford_qa_and_vectors_pickled()
    print("Loading successful.")
    return word2vec, df, text_vectors, text_words, question_data


def get_word2vec_and_stanford_qa_from_scratch():
    """
    Master loading and processing function.
    :return:
        word2vec -- Gensim word2vec model
        df -- Original dataframe as processed in Bharath's kernel here:
            https://www.kaggle.com/bharathsh/stanford-q-a-json-to-clean-dataframe
            -- with a few additional data post-processing corrections
        text_vectors -- dictionary where key is a context id, value is a matrix of
            word2vec embeddings corresponding to that text passage
        text_words -- dictionary where key is a context id, value is a list of strings.
            each string is a word corresponding to a row in the vector matrix, and
            that row contains the vector for that word in that position in the passage
            meant for aligning model guesses with actual target labels for qualitative
            examination of results
        question_data -- a row of dictionaries
            (can be easily converted to dataframe with
                `pd.DataFrame.from_records(question_data)` )

            Each dictionary contains the following keys:

            idx: corresponding row index in Stanford dataframe.
                (This should also equal the index of this row in the list of rows returned)
            c_id: context_id corresponding to the text passage this question is based on.
                can be used to look up the corresponding vectorizations and wordlists in
                text_vectors and text_words.

            question: original unprocessed text of the question

            text: original, unprocessed text of the context passage,
            answer: original, unprocessed text of the answer

            question_matrix: vectorization of the question, one row for each word for which
                a vector representation was found
            answer_vector: an array of target flags corresponding to each word in the context
                passage wordlist and each vector in the text matrix: 1 if the corresponding
                word is part of the answer, 0 otherwise

    """
    word2vec = get_word2vec_model()
    df = import_stanford_qa_data()
    text_vectors, text_words, question_data = get_text_and_question_vectors(
        df, word2vec)
    return word2vec, df, text_vectors, text_words, question_data


def get_word2vec_model(filename=WORD2VEC_PATH):
    # for purposes of running on kaggle kernel where
    # word2vec is not available, create a mock
    # instance that returns small zero vectors
    # for testing purposes

    if not KAGGLE_DEMO:
        model = KeyedVectors.load_word2vec_format(
            filename, binary=True)
    else:
        class MockModel(object):
            def __init__(self, *args, **kwargs):
                pass

            def get_vector(self, *args, **kwargs):
                return np.zeros((5,))

        model = MockModel()

    return model


def save_vectors_and_df(question_data, text_vectors, text_words, df):
    save_questions_and_text_vectors(
        question_data, text_vectors, text_words)
    pickle_me(df, DF_PICKLE)


def get_stanford_qa_and_vectors_pickled():
    """
    load saved question data and vector representations.  since original dataframe
    may not be needed anymore, optionally just call `load_questions_and_text_vectors()`
    directly

    """
    df = unpickle_me(DF_PICKLE)
    question_vectors, text_vectors, text_words = load_questions_and_text_vectors()
    return df, question_vectors, text_vectors, text_words


def save_questions_and_text_vectors(question_data, text_vectors, text_words):
    """
    just a simple function to pickle the final results for later use --
    possibly replace this with a better / safer storage protocol than pickle later
    """
    pickle_me(question_data, QUESTION_DATA_PICKLE)
    pickle_me(text_vectors, TEXT_VECTOR_PICKLE)
    pickle_me(text_words, TEXT_WORDS_PICKLE)


def load_questions_and_text_vectors():
    """
    restore pre-processed question data, text vectorization,
    and wordlists from disk
    """
    question_data = unpickle_me(QUESTION_DATA_PICKLE)
    text_vectors = unpickle_me(TEXT_VECTOR_PICKLE)
    text_words = unpickle_me(TEXT_WORDS_PICKLE)
    return question_data, text_vectors, text_words


def pickle_me(obj, filename):
    with open(filename, 'wb') as fh:
        pickle.dump(obj, fh)


def unpickle_me(filename):
    with open(filename, 'rb') as fh:
        return pickle.load(fh)


def import_stanford_qa_data():
    # thanks to Bharath, https://www.kaggle.com/bharathsh/stanford-q-a-json-to-clean-dataframe,
    # for code adapted for this use
    #
    train_path = os.path.join(STANFORD_DATA_PATH, 'train-v1.1.json')
    path = ['data', 'paragraphs', 'qas', 'answers']
    with open(train_path) as fh:
        raw_json = json.loads(fh.read())

    js = pd.io.json.json_normalize(raw_json, path)
    m = pd.io.json.json_normalize(raw_json, path[:-1])
    r = pd.io.json.json_normalize(raw_json, path[:-2])

    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx = np.repeat(m['id'].values, m['answers'].str.len())
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
    """
    some specific corrections -- plus some manual copy/paste issues make certain
    answers hard to parse

    TODO: possibly just save / reupload data with these corrections made
    """
    for idx in range(df.shape[0]):
        answer = df.loc[idx, 'text']
        if answer.startswith(". ") or answer.startswith(", "):
            df.loc[idx, 'text'] = answer[2:]
            df.loc[idx, 'answer_start'] += 2


    for _ in range(0, 5):
        df = df.drop(16818+_)

    df = df.reset_index(drop=True)

    for _ in range(0, 4):
        df = df.drop(38416 + _)

    df = df.reset_index(drop=True)
    # person could not come up with questions and entered junk data
    df = df.drop(38537)

    df = df.reset_index(drop=True)
    df = df.drop(38666)

    df = df.reset_index(drop=True)
    df = df.drop(38693)

    df = df.reset_index(drop=True)

    df.loc[1590, 'answer_start'] = 332
    df.loc[1590, 'text'] = 'seven'

    df.loc[1595, 'answer_start'] = 332
    df.loc[1595, 'text'] = 'seven'

    df.loc[1600, 'answer_start'] = 332
    df.loc[1600, 'text'] = 'seven'

    df.loc[2478, 'answer_start'] = 302
    df.loc[2478, 'text'] = 'three'

    df.loc[2491, 'answer_start'] = 60
    df.loc[2491, 'text'] = "the media player included with the iPhone and iPad, a combination of the Music and Videos apps"

    df.loc[2501, 'answer_start'] = 396


    df.loc[3215, 'answer_start'] = 242
    df.loc[3218, 'answer_start'] = 242

    df.loc[3472, 'answer_start'] = 242

    df.loc[3490, 'text'] = "a million"
    df.loc[3490, 'answer_start'] = 268



    df.loc[4051, 'text'] = "over 50"
    df.loc[4051, 'answer_start'] = 327

    df.loc[4168, 'text'] = "92%"
    df.loc[4168, 'answer_start'] = 444

    df.loc[4182, 'answer_start'] = 618


    df.loc[5121, 'text'] = "15 km southwest of Dushanbe"
    df.loc[5121, 'answer_start'] = 499


    df.loc[5919, 'text'] = "seventh"
    df.loc[5919, 'answer_start'] = 983

    df.loc[5924, 'text'] = "seventh"
    df.loc[5924, 'answer_start'] = 983


    df.loc[5990, 'text'] = "three"
    df.loc[5990, 'answer_start'] = 925


    df.loc[5992, 'text'] = "three"
    df.loc[5992, 'answer_start'] = 925


    df.loc[6314, 'text'] = "six"
    df.loc[6314, 'answer_start'] = 292

    df.loc[6316, 'text'] = "six"
    df.loc[6316, 'answer_start'] = 13

    df.loc[6735, 'answer_start'] = 118

    df.loc[6769, 'answer_start'] = 513
    df.loc[6769, 'text'] = "ritual, visualization, physical exercises, and meditation"

    df.loc[6985, 'answer_start'] = 333
    df.loc[6985, 'text'] = "fifteenth"

    df.loc[7528, 'answer_start'] = 39
    df.loc[7528, 'text'] = "fifteenth"

    df.loc[7529, 'answer_start'] = 326
    df.loc[7529, 'text'] = "fifth"

    df.loc[7622, 'answer_start'] = 411
    df.loc[7622, 'text'] = "fifteenth"

    df.loc[7625, 'answer_start'] = 388
    df.loc[7625, 'text'] = "almost a decade, from 2003 to 2012"

    df.loc[7626, 'answer_start'] = 148
    df.loc[7626, 'text'] = "eight"

    df.loc[7676, 'answer_start'] = 87
    df.loc[7676, 'text'] = "nine"

    df.loc[7676, 'answer_start'] = 113
    df.loc[7676, 'text'] = "Director Bruce Gower won a Primetime Emmy Award for Outstanding Directing For A Variety, Music Or Comedy Series in 2009, and the show won a Creative Arts Emmys each in 2007 and 2008, three in 2009, and two in 2011"
    df.loc[7678, 'answer_start'] = 113
    df.loc[7678, 'text'] = "Director Bruce Gower won a Primetime Emmy Award for Outstanding Directing For A Variety, Music Or Comedy Series in 2009, and the show won a Creative Arts Emmys each in 2007 and 2008, three in 2009, and two in 2011"

    df.loc[8189, 'answer_start'] = 310

    df.loc[8735, 'answer_start'] = 137
    df.loc[8735, 'text'] = "its synthesis of and reaction to the world around it"


    df.loc[8835, 'answer_start'] = 373
    df.loc[8835, 'text'] = "300 of which was paid by Cambridge University Press, 200 by the Royal Society of London, and 50 apiece by Whitehead and Russell themselves"


    df.loc[9745, 'answer_start'] = 621


    df.loc[10227, 'answer_start'] = 81

    df.loc[10354, 'answer_start'] = 326
    df.loc[10354, 'text'] = 'one regiment each of artillery, armour, and combat engineers'


    df.loc[11780, 'answer_start'] = 239
    df.loc[11780, 'text'] = '30'

    df.loc[12189, 'answer_start'] = 96
    df.loc[12189, 'text'] = 'two'

    df.loc[12874, 'answer_start'] = 154
    df.loc[12874, 'text'] = 'five'


    df.loc[13060, 'answer_start'] = 272

    df.loc[13232, 'answer_start'] = 505


    df.loc[13787, 'answer_start'] = 362
    df.loc[13787, 'text'] = 'nine'


    df.loc[13791, 'answer_start'] = 189
    df.loc[13791, 'text'] = 'three'

    df.loc[13799, 'answer_start'] = 944
    df.loc[13799, 'text'] = 'four'

    df.loc[13863, 'answer_start'] = 185
    df.loc[13863, 'text'] = 'I, II and III'

    df.loc[14377, 'answer_start'] = 65
    df.loc[14377, 'text'] = 'one'

    df.loc[14448, 'answer_start'] = 78
    df.loc[14448, 'text'] = 'three'


    df.loc[14473, 'answer_start'] = 259
    df.loc[14473, 'text'] = '⟨p⟩'


    df.loc[14516, 'answer_start'] = 341

    df.loc[14582, 'answer_start'] = 12
    df.loc[14582, 'text'] = 'two'

    df.loc[14589, 'answer_start'] = 674
    df.loc[14589, 'text'] = "partial negative"



    df.loc[15159, 'answer_start'] = 108

    df.loc[15246, 'answer_start'] = 186

    df.loc[15865, 'text'] = 'i'
    df.loc[15865, 'answer_start'] = 231

    df.loc[15951, 'text'] = 'one'
    df.loc[15951, 'answer_start'] = 114


    df.loc[16124, 'answer_start'] = 85

    df.loc[16297, 'answer_start'] = 121

    df.loc[17106, 'answer_start'] = 200
    df.loc[17111, 'answer_start'] = 413


    df.loc[17168, 'text'] = 'eight'
    df.loc[17168, 'answer_start'] = 63

    df.loc[17280, 'answer_start'] = 151

    df.loc[18256, 'answer_start'] = 228


    df.loc[18369, 'answer_start'] = 370
    df.loc[18369, 'text'] = 'ser'

    df.loc[18707, 'answer_start'] = 0

    df.loc[23848, 'answer_start'] = 3
    df.loc[23848, 'text'] = 'the 1950s'

    df.loc[25521, 'answer_start'] = 418
    df.loc[25521, 'text'] = 'sync word'
    ##
    df.loc[27972, 'answer_start'] = 111
    df.loc[27972, 'text'] = 'the difficulty of the elements the gymnast attempts and whether or not the gymnast meets composition requirements'
    df.loc[27972, 'question'] = 'How is the start value determined?'

    df.loc[28010, 'answer_start'] = 116

    df.loc[28010, 'text'] ='external force which the gymnasts have to overcome with their muscle force and has an impact on the gymnasts linear and angular momentum'


    df.loc[34600, 'answer_start'] = 404
    df.loc[34600, 'text'] = 'Men did not show any sexual arousal to non-human visual stimuli'

    df.loc[36071, 'answer_start'] = 222
    df.loc[36071, 'text'] = 'the information must be changed'


    df.loc[39349, 'answer_start'] = 40
    df.loc[39349, 'text'] = 'a US$5 million grant for the International Justice Mission (IJM)'

    df.loc[45018, 'answer_start'] = 454
    df.loc[45018, 'text'] = 'because Florida had become "a derelict open to the occupancy of every enemy, civilized or savage, of the United States, and serving no other earthly purpose than as a post of annoyance to them.'
    # df.loc[38536, 'answer_start'] = 491
    # df.loc[38536, 'question'] = 'How long did the Rhodians hold out under siege by Demetrius Poliorcetes?'
    # df.loc[38536, 'text'] = 'one year'


    df.loc[45076, 'answer_start'] = 562
    df.loc[45076, 'text'] = 'the first post-Reconstruction Republican governor'

    df.loc[45077, 'answer_start'] = 703
    df.loc[45077, 'text'] = "the state's first post-reconstruction Republican US Senator"

    df.loc[47651, 'answer_start'] = 208
    df.loc[47651, 'text'] = 'increasing numbers of airlines have began launching direct flights'



    df.loc[52335, 'answer_start'] = 147
    df.loc[52335, 'text'] = 'Mahmoud Massahi'


    df.loc[53520, 'answer_start'] = 197
    df.loc[53520, 'question'] = 'How many days after the Soviet Union issued their ultimatum did the Romanians meet their demands?'
    df.loc[53520, 'text'] = 'Two'



    df.loc[55205, 'answer_start'] = 48
    df.loc[55205, 'text'] = 'the rebuilt Wembley Stadium'

    df.loc[55752, 'answer_start'] = 370
    df.loc[55752, 'text'] = 'lower-elevation areas of the Piedmont'


    df.loc[56306, 'answer_start'] = 506
    df.loc[56306, 'text'] = 'the wings of flightless birds and the rudiments of pelvis and leg bones found in some snakes'

    df.loc[57712, 'answer_start'] = 78
    df.loc[57712, 'text'] = 'vinyl'

    df.loc[58754, 'answer_start'] = 35
    df.loc[58754, 'text'] = 'the evolving relationship between state governments and the federal government of the United States'

    df.loc[58813, 'answer_start'] = 239
    df.loc[58813, 'question'] = 'How is public spending distributed in Spain?'
    df.loc[58813, 'text'] = 'the central government accounting for just 18% of public spending, 38% for the regional governments, 13% for the local councils, and the remaining 31% for the social security system'

    df.loc[61893, 'answer_start'] = 101
    df.loc[61893, 'text'] = 'Little'

    df.loc[66861, 'answer_start'] = 520
    df.loc[66861, 'text'] = 'parks, schools, public buildings, proper roads and the other amenities that characterise a modern city'

    df.loc[68312, 'answer_start'] = 29
    df.loc[68312, 'question'] = 'What occurs with osmotic diarrhea?'
    df.loc[68312, 'text'] = 'too much water is drawn into the bowels'


    df.loc[70195, 'answer_start'] = 427
    df.loc[70195, 'questions'] = "What did the Native American tribes fail to accomplish in the later Pontiac's War?"
    df.loc[70195, 'text'] = 'returning them to their pre-war status'


    df.loc[70443, 'answer_start'] = 495
    df.loc[70443, 'text'] = 'somehow-belligerent'


    df.loc[71357, 'answer_start'] = 118
    df.loc[71357, 'text'] = 'science-fiction and adventure'

    df.loc[75502, 'text'] = '140 million'
    df.loc[75502, 'answer_start'] =389

    df.loc[85016, 'answer_start'] = 387

    df.loc[85004, 'question'] = "What was Greece's reference year budget deficit?"
    df.loc[85004, 'text'] = '3.38% of GDP'
    df.loc[85004, 'answer_start'] = 385

    df.loc[85016, 'text'] = "this affected deficit values after 2001 (when Greece had already been admitted into the Eurozone)"
    df.loc[85016, 'answer_start'] = 363


    df.loc[87208, 'text'] = "typically discarded"
    df.loc[87208, 'answer_start'] = 216


    return df


def get_text_and_question_vectors(df, word2vec):
    question_vectors = []
    text_vectors = dict()
    text_words_dict = dict()

    df['text_matrix'] = None
    df['question_matrix'] = None
    df['answer_vector'] = None
    n_rows = df.shape[0]
    total_pos = 0
    total_neg = 0
    # for idx in range(n_rows):
    #     print(idx)
    for idx in range(0, n_rows):  # range(n_rows):
        if not idx % 250:
            print("{}/{}".format(idx, n_rows))
        question, context, answer_start, answer_text, c_id = extract_fields(df, idx)
        # print((idx, question, answer_start, answer_text))
        # especially annoying special cases
        if answer_text != '.50-inch' and o_point_rgx.match(answer_text):
            answer_text = "0" + answer_text
            answer_start -= 1

        question_matrix, question_words = text_to_matrix(
            text=question, model=word2vec)
        text_matrix, text_words, answer_vec, num_pos, num_neg = text_to_matrix_with_answer(
            text=context, answer=answer_text,
            answer_idx=answer_start, model=word2vec)

        text_words_dict[c_id] = text_words

        row = dict(
            idx=idx,
            c_id=c_id,
            question=question,
            text=context,
            answer=answer_text,
            question_matrix=question_matrix,
            answer_vector=answer_vec, )
        question_vectors.append(row)
        text_vectors[c_id] = text_matrix
        total_pos += num_pos
        total_neg += num_neg

    print(
        "On average there were {} answer words and {} other words per row."
        "\nPositive should be weighted about {}.".format(
            total_pos / n_rows, total_neg / n_rows, total_neg / total_pos))
    return text_vectors, text_words_dict, question_vectors


def extract_fields(df, idx):
    row = df.iloc[idx]
    return row.loc[[
        'question', 'context', 'answer_start', 'text', 'c_id']]


def text_to_matrix(text, model):
    """
    general purpose vectorize a text -- used for the
    question part currently, as the context vectorization is
    more complex and needs to keep track of answer
    """
    words = split_with_dollarsign(text)
    vectors = []
    remaining_words = []
    idx, end = 0, len(words)
    clean_words = list(map(clean_for_w2v, words))
    while idx < end:
        word = clean_words[idx]
        try:
            possible_next = clean_words[idx + 1]
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
    """
    split out a text and find vectors for each lexical item
    also keeps track of target Y values for each lexical item
    in the text: 0 if the word is not part of the answer, 1 if it is

    There is a lot of redundancy here since every time the text is
    reused for a different question, the vectorization is repeated
    since the answer positioning is intertwined with the vector
    search.  Lots of potential to improve, but since it runs in
    a couple minutes on my machine and I can save results, will wait
    for a complete overhaul in possible future versions

    """
    # first remove punctuation
    text_words = split_with_dollarsign(text)
    answer_words = split_with_dollarsign(answer)
    in_answer = False
    num_answer_words = len(answer_words)
    answer_words_found = 0
    vectors = []
    answer_flags = []
    remaining_words = []
    curr_char_idx = 0

    for n, word in enumerate(text_words):
        if curr_char_idx == answer_idx or (
                word and curr_char_idx <= answer_idx < (curr_char_idx + (len(word) or 1))):
            # print((word, curr_char_idx, answer_idx))
            if answer_idx != curr_char_idx:
                num_off = np.abs(answer_idx - curr_char_idx)
                off_str = word[:num_off]
                print("Watch out, was {} off because of {}".format(
                    num_off, off_str))
                print(text)
                print(text_words)
                print(answer_words)
                print(answer)
                if off_str not in ["US", "“", "‘", "-", "−", "⟨", " "] and off_str not in string.punctuation:
                    raise Exception
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
    if np.sum(answer_flags) == 0:
        print(answer_words_found)
        print(answer_flags)
        print("Answer: {}".format(answer))
        print("Text: {}".format(text))
        print(answer_words)
        print(text_words)
        raise ValueError("Answer words not found in word2vec!")
    return (
        matrix,
        remaining_words,
        answer_flags,
        answer_words_found,
        len(remaining_words) - answer_words_found)


def clean_for_w2v(word):
    """
    Do some basic universal cleaning to make the word findable in word2vec.
    Most punctuation needs to be purged, numerals need to be handled, but
    some meaningful punctuation such as currency symbols needs to be preserved.

    so far haven't been able to perfectly emulate google's handling of certain compound
    terms: tennis-player -> tennis_player and other similar, so they are mostly handled
    by pre-splitting and separate vector representations.
    """
    rtn = ""
    word = replace_digits(word)
    for ch in word:
        if ch.isalnum():
            rtn += ch
        elif ch in '$£¥€#':
            # keep numeral symbol # and currency symbols as vectors exist
            rtn += ch
        elif ch in string.punctuation:
            pass

    return rtn


def replace_digits(word):
    """
    google word2vec seems to have handled digits like so:
    single digits are preserved as-is: 1,2,3 etc. have their own
    vectors represented.  Two or more digits are masked as '#' for
    each digit, and strings of '##' of varying lengths have vector
    representations.  Convert the incoming strings to this representation
    to look up the appropriate vectorization.
    """
    new_str = str(word)
    matches = replace_digits_rgx.finditer(word)
    for m in matches:
        new_str = new_str[:m.start(0)] + "#" * len(m[0]) + new_str[m.end(0):]
    return new_str


def sep_digits(word):
    """
    occasionally after swapping out numerals for '#', we may end up with a word that
    is mixed letters and ##s.  break out the alphabetic part and the numeric part and
    return the pieces in the order we foudn them
    """
    first = ""
    second = ""
    dig_first = None
    for n, ch in enumerate(word):
        if not n:
            dig_first = ch == '#'

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


def get_vec(word, model, possible_next=None):
    """
    elaborate function to find best vector(s) for a given word in google word2vec,
    accounting for case sensitivity inconsistency, some english/GB spelling differences,
    parsing issues, unicode issues, etc.

    hacky but works, would be really nice to refactor entire text processing chain

    possible TODO: implement parsing of a 'possible next word'
        (e.g. model tries 'Taj_Mahal' before just returning vector for 'Taj')

    returns a list of vectors in order
        a list of strings in order (to calculate the distance moved along the text)
        an additional character offset (to account for characters lost in splitting
            and other parsing operations to allow the text processor to keept track
            of where the answer is) --

    """
    if not word:
        return [None], [None], 1

    # hack here -- since $ and other currency symbols have
    # to be split by themselves in order to keep the character counter
    # from advancing an extra character every time a $
    # is seen, it has to return 0 for the spacing increment
    # in that case.
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

                    for k, v in suffix_replacements.items():
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
            return possible_combined, [possible_combined], 2
    else:
        return _get_vec(word)


if __name__ == '__main__':
    main()
