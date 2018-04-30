from question_answer.util import pickle_me, unpickle_me
from question_answer import process_data
from keras.models import load_model
from question_answer.util import add_dim, sigmoid
import numpy as np

MODEL_PATH = "C:/Users/Joseph Giroux/Datasets/qa_model.h5"

def save_qa_model(model):
    model.save(MODEL_PATH)

def load_qa_model():
    return load_model(MODEL_PATH)




def get_train_test_valid_groups(vectors):
    total = 87587
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
                # print(answers[n].shape)
                answers[n] = np.concatenate((
                    np.zeros((pre_pad, 2)),
                    answers[n],
                    np.zeros((post_pad, 2))))
                # print(answers[n].shape)
                # answers[n] += pre_pad
                # sparse encoding, more trouble than its worth
                assert answers[n].shape[0] == desired_len

    return vectors, answers



def answer_start_end(answer_vector):
    answer_span = np.nonzero(answer_vector)

    answer_start = np.zeros(shape=answer_vector.shape)
    answer_start[np.min(answer_span)] = 1

    answer_end = np.zeros(shape=answer_vector.shape)
    answer_end[np.max(answer_span)] = 1
    # print(answer_start)
    # print(answer_start.shape)
    # print(answer_end)
    return answer_start, answer_end



def batch_generator(
        question_vectors, text_vectors,
        batch_size, randomize=False,
        data_size=60000, always=None):

    while True:
        for start in range(0, data_size, batch_size):
            end = start + batch_size
            text_inputs = []
            question_inputs = []
            answer_vectors = []
            for row in question_vectors[start:end]:
                if always:
                    row = question_vectors[always]
                elif randomize:
                    row = question_vectors[int(np.random.randint(data_size))]
                text_inputs.append(text_vectors[row['c_id']])
                question_inputs.append(row['question_matrix'])
                ans = np.stack(answer_start_end(row['answer_vector']), axis=-1)
                answer_vectors.append(ans)


            text_inputs, answer_vectors = pad_vectors(
                text_inputs, answers=answer_vectors)
            # print('ITS ALWAYS THE WRONG SHAPE')
            # print(answer_vectors.shape)
            question_inputs, _ = pad_vectors(
                question_inputs)
            # print(answer_vectors[0].shape)
            # print(np.stack(answer_vectors).shape)
            #
            # print(np.stack(answer_vectors).shape)
            rtn = (
                [np.stack(text_inputs),
                 np.stack(question_inputs)],
                np.stack(answer_vectors))

            yield rtn


def show_example(
        model, df,
        question_vectors,
        text_vectors,
        text_words,
        idx=None):

    if idx is None:
        idx = np.random.randint(0, df.shape[0])

    print(question_vectors[idx]['question'])
    print(question_vectors[idx]['answer'])
    question, context, answer_start, answer_text, c_id = process_data.extract_fields(df, idx)
    question_vector = question_vectors[idx]['question_matrix']
    answer_vector = question_vectors[idx]['answer_vector']
    # answer_start, answer_end = answer_start_end(question_vectors[idx]['answer_vector'])

    text_vector = text_vectors[c_id]
    pred = model.predict([add_dim(text_vector), add_dim(question_vector)])

    def display_guess(matrix, words):
        # print("LOGITS SHAPE")
        # print(matrix.shape)

        sums = np.sum(matrix, axis=0)
        start_end = np.argmax(matrix, axis=1)
        guess_start_idx = start_end[0][0]
        guess_end_idx = start_end[0][1]

        for n, word in enumerate(words):

            try:
                pass
            except ValueError:
                bar = "NaN"
            print("( {answer} ({guess}) ) {word}: [{start_score} -> {end_score}]".format(
                answer="YES" if answer_vector[n] else "NO",
                guess={guess_start_idx: 'start', guess_end_idx: 'end'}[n] if n in (
                    guess_start_idx, guess_end_idx) else '--',
                word=word, start_score=matrix[0][n][0],
                end_score=matrix[0][n][1],))

    display_guess(pred, text_words[c_id])
