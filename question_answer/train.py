from question_answer.util import pickle_me, unpickle_me
from keras.models import load_model
from question_answer.util import add_dim

MODEL_PATH = "C:/Users/Joseph Giroux/Datasets/qa_model.h5"

def save_qa_model(model):
    model.save(MODEL_PATH)

def load_qa_model():
    return load_model(MODEL_PATH)