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
            train_question_x.append(vectors[n])
            train_y.append(add_dim(add_dim(vectors[n]["answer_vector"])))
        elif group == test:
            test_question_x.append(vectors[n])
            test_y.append(add_dim(add_dim(vectors[n]["answer_vector"])))
        elif group == valid:
            valid_question_x.append(vectors[n])
            valid_y.append(add_dim(add_dim(vectors[n]["answer_vector"])))
        elif group == final_valid:
            final_valid_question_x.append(vectors[n])
            final_valid_y.append(add_dim(add_dim(vectors[n]["answer_vector"])))

    all_train = (train_question_x, train_y)
    all_test = (test_question_x, test_y)
    all_valid = (valid_question_x, valid_y)
    all_final_valid = (final_valid_question_x, final_valid_y)
    return (all_train, all_test, all_valid, all_final_valid)




def pad_vectors(vectors, desired_len=None, n_features=300, answers=None):
    if desired_len is None:
        if not vectors:
            return vectors
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


def batch_generator(
        question_vectors, text_vectors,
        batch_size, randomize=False, data_size=60000):

    if randomize:
        random_options = list(range(0, data_size))
    while True:
        for start in range(0, data_size, batch_size):
            end = start + batch_size
            text_inputs = []
            question_inputs = []
            answer_vectors = []
            for row in question_vectors[start:end]:
                if randomize:
                    row = question_vectors[int(np.random.choice(random_options))]
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


