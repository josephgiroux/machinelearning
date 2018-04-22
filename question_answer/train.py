from question_answer.util import pickle_me, unpickle_me
from keras.models import load_model
from question_answer.util import add_dim
import numpy as np

MODEL_PATH = "C:/Users/Joseph Giroux/Datasets/qa_model.h5"

def save_qa_model(model):
    model.save(MODEL_PATH)

def load_qa_model():
    return load_model(MODEL_PATH)


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

            yield ([np.stack(text_inputs), np.stack(question_inputs)], np.stack(answer_vectors))



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

            yield ([np.stack(text_inputs), np.stack(question_inputs)], np.stack(answer_vectors))

