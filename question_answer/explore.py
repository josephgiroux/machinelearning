from importlib import reload
from question_answer.process_data import *
import question_answer
from question_answer.network import combined_network
reload(question_answer.network)
from question_answer.network import combined_network
from question_answer.util import *
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

# On average there were 3.366419707987534 answer words and 105.92495348120413 other words per row.
# Positive should be weighted about 31.465165567405347.

df = pickle_me(df, "C:/Users/Joseph Giroux/Datasets/df.pkl")