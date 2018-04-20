from importlib import reload
from question_answer.process_data import *
import question_answer
from question_answer.network import combined_network
reload(question_answer.network)
from question_answer.network import combined_network

import question_answer.process_data as p
reload(question_answer.process_data)
reload(p)
reload(question_answer)
import question_answer.process_data as p

# from question_answer.process_data import *

model = combined_network()
word2vec, df = get_word2vec_and_stanford_qa()
pred = p.one_question_test(df, word2vec, model)


def add_vector_information(df, word2vec):
    df['text_matrix'] = None
    df['question_matrix'] = None
    df['answer_vector'] = None
    n_rows = df.shape[0]
    print(df.shape[0])
    total_pos = 0
    total_neg = 0
    # for idx in range(n_rows):
    #     print(idx)
    for idx in range(n_rows):
        print(idx)
        question, context, answer_start, answer_text = extract_fields(df, idx)

        question_mat, question_words = text_to_matrix(
            text=question, model=word2vec)

        text_mat, text_words, answer_vec, num_pos, num_neg = text_to_matrix_with_answer(
            text=context, answer=answer_text,
            answer_idx=answer_start, model=word2vec)

        df.iloc[idx].at['text_matrix'] = text_mat
        df.iloc[idx].at['question_matrix'] = question_mat
        df.iloc[idx].at['answer_vector'] = answer_vec
        total_pos += num_pos
        total_neg += num_neg

    print(
        "On average there were {} answer words and {} other words per row."
        "\nPositive should be weighted about {}.".format(
            total_pos / n_rows, total_neg / n_rows, total_neg / total_pos))
    return df

df = add_vector_information(df, word2vec)

