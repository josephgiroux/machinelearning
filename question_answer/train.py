from question_answer.util import pickle_me, unpickle_me
from question_answer import process_data
from keras.models import load_model
from question_answer.util import add_dim
import numpy as np

MODEL_PATH = "C:/Users/Joseph Giroux/Datasets/qa_model.h5"

def save_qa_model(model):
    model.save(MODEL_PATH)

def load_qa_model():
    return load_model(MODEL_PATH)



def pad_vectors(vectors, desired_len=None, n_features=300, answers=None):
    if desired_len is None:
        desired_len = max(vectors, key=lambda v: v.shape[0]).shape[0]
        # print("Desired len:", desired_len)


    for n, vec in enumerate(vectors):

        diff = desired_len - vec.shape[0]
        # print("Diff: ", diff)
        if diff < 0:
            raise ValueError("Can't pad to a shorter length")

        if diff:
            post_pad = diff // 2
            pre_pad = diff - post_pad

            new_matrix = np.concatenate([np.zeros((1, n_features))] * pre_pad + [vec] + (
                    [np.zeros((1, n_features))] * post_pad))
            vectors[n] = new_matrix

            assert new_matrix.shape[0] == desired_len

            if answers is not None:
                answers[n] = np.concatenate((
                    np.zeros((1, pre_pad)),
                    answers[n],
                    np.zeros((1, post_pad))), axis=-1)

    return vectors, answers


def batch_generator(question_vectors, text_vectors, batch_size):

    while True:
        for start in range(0, 60000, batch_size):
            end = start + batch_size
            text_inputs = []
            question_inputs = []
            answer_vectors = []
            for row in question_vectors[start:end]:
                text_inputs.append(text_vectors[row['c_id']])
                question_inputs.append(row['question_matrix'])
                answer_vectors.append(add_dim(row['answer_vector']))


            text_inputs, answer_vectors = pad_vectors(
                text_inputs, answers=answer_vectors)
            question_inputs, _ = pad_vectors(
                question_inputs)

            yield (
                [np.stack(text_inputs), np.stack(question_inputs)],
                np.stack(answer_vectors))



def valid_generator(question_vectors, text_vectors, batch_size):

    while True:
        for start in range(60000, 70000, batch_size):
            end = start + batch_size
            text_inputs = []
            question_inputs = []
            answer_vectors = []
            for row in question_vectors[start:end]:
                text_inputs.append(text_vectors[row['c_id']])
                question_inputs.append(row['question_matrix'])
                answer_vectors.append(add_dim(row['answer_vector']))


            text_inputs, answer_vectors = pad_vectors(
                text_inputs, answers=answer_vectors)
            question_inputs, _ = pad_vectors(
                question_inputs)

            yield (
                [np.stack(text_inputs), np.stack(question_inputs)],
                np.stack(answer_vectors))



