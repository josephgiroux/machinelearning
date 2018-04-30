




from importlib import reload
from question_answer.process_data import *
from question_answer.train import *
import question_answer
from question_answer.network import combined_network_one_reader
from question_answer.util import *
import question_answer.process_data as p

# from question_answer.process_data import *

# model = combined_network()
# word2vec, df, vectors = get_word2vec_and_stanford_qa_from_scratch()
# pred = p.one_question_test(df, word2vec, model)


# vectors = get_vector_information(df, word2vec)
# save_vectors_and_df(vectors=vectors, df=df)
# word2vec, df, text_vectors, text_words, question_vectors = get_word2vec_and_stanford_qa_from_scratch()
#
# save_questions_and_text_vectors(question_vectors, text_vectors, text_words)

df, question_data, text_vectors, text_words = get_stanford_qa_and_vectors_pickled()


(all_train, all_test, all_valid, all_final_valid) = get_train_test_valid_groups(question_data)

train_question_x, train_y = all_train
test_question_x, test_y = all_test


model = combined_network_one_reader(
    lr=0.001, pos_weight=35)
model.summary()


show_example(
        model, df,
        question_data,
        text_vectors,
        text_words,
        idx=1)




#
# for layer in model.layers:
#     weights = layer.get_weights()
#     for w in weights:
#         print(w)

model.fit_generator(
    generator=batch_generator(
        text_vectors=text_vectors,
        question_vectors=train_question_x,
        batch_size=40, data_size=2,
        randomize=True),
    validation_data=batch_generator(
        text_vectors=text_vectors,
        question_vectors=test_question_x,
        batch_size=40, data_size=10000),
    steps_per_epoch=50,
    validation_steps=25)



(all_train, all_test, all_valid, all_final_valid) = get_train_test_valid_groups(question_data)

train_text_x, train_question_x, train_y = all_train

steps_per_epoch = 1500

gen = batch_generator(
        text_vectors=text_vectors,
        question_vectors=train_question_x,
        batch_size=40, data_size=40,
        randomize=True)


x = next(gen)



len(vectors)
# On average there were 3.366419707987534 answer words and 105.92495348120413 other words per row.
# Positive should be weighted about 31.465165567405347.

# df = pickle_me(df, "C:/Users/Joseph Giroux/Datasets/df.pkl")