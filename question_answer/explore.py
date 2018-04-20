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


df = add_vector_information(df, word2vec)

