from importlib import reload
from question_answer.process_data import *
import question_answer
from question_answer.network import combined_network
reload(question_answer.network)
from question_answer.network import combined_network

word2vec, df = get_word2vec_and_stanford_qa()
model = combined_network()

pred = one_question_test(df, word2vec, model)