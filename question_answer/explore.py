




from importlib import reload
from question_answer.process_data import *
from question_answer.train import *
import question_answer
from question_answer.network import combined_network
reload(question_answer.network)
from question_answer.network import combined_network
from question_answer.util import *
import question_answer.process_data as p

# from question_answer.process_data import *

# model = combined_network()
# word2vec, df, vectors = get_word2vec_and_stanford_qa_from_scratch()
# pred = p.one_question_test(df, word2vec, model)


# vectors = get_vector_information(df, word2vec)
# save_vectors_and_df(vectors=vectors, df=df)
# word2vec, df, text_vectors, question_vectors = get_word2vec_and_stanford_qa_from_scratch()

# save_questions_and_text_vectors(question_vectors, text_vectors)

df, question_vectors, text_vectors = get_stanford_qa_and_vectors_pickled()

gen = batch_generator(
        text_vectors=text_vectors,
        question_vectors=question_vectors,
        batch_size=40, randomize=True)
x = next(gen)



model = combined_network(lr=0.01)


(all_train, all_test, all_valid, all_final_valid) = get_train_test_valid_groups(question_vectors)

train_question_x, train_y = all_train
test_question_x, test_y = all_test


model.fit_generator(
    generator=batch_generator(
        text_vectors=text_vectors,
        question_vectors=train_question_x,
        batch_size=40, data_size=60000,
        randomize=True),
    validation_data=batch_generator(
        text_vectors=text_vectors,
        question_vectors=test_question_x,
        batch_size=40, data_size=10000),
    steps_per_epoch=1500,
    validation_steps=250)




(all_train, all_test, all_valid, all_final_valid) = get_train_test_valid_groups(vectors)

train_text_x, train_question_x, train_y = all_train

steps_per_epoch = 1500


x = next(gen)

print(np.repeat(train_text_x[1], 32, axis=0).shape)

print(np.repeat(train_question_x[1], 32, axis=0).shape)
print(np.repeat(train_y[1], 32, axis=0).shape)


total_loss = 0.0
rounds = 0
train_size = 10000
indices = list(range(60000))
network = combined_network()


for _ in range(1):
    n = np.random.choice(indices)
    n = 1
    h = model.fit(
        x=[np.repeat(train_text_x[n], 32, axis=0),
           np.repeat(train_question_x[n], 32, axis=0)],
        y=np.repeat(train_y[n], 32, axis=0),
        batch_size=32)

    total_loss += h.history["loss"][0]
    rounds += 1
    print(total_loss/rounds)

print(h.history)

start = 0
end = 100


text_batch = train_text_x[start:end]
question_batch = train_question_x[start:end]
y_batch = train_y[start:end]



print(text_batch[0].shape)
print(question_batch[0].shape)
print(y_batch[0].shape)


total_loss = 0.0
rounds = 0
for n in range(len(text_batch)):
    h = model.fit(
        x=[text_batch[n], question_batch[n]],
        y=y_batch[n])
    total_loss += h.history["loss"][0]
    rounds += 1
    print(total_loss/rounds)

print(h.history)



print(np.repeat(train_text_x[1], 32, axis=0).shape)

print(np.repeat(train_question_x[1], 32, axis=0).shape)
print(np.repeat(train_y[1], 32, axis=0).shape)



h = model.fit(
        x=[text_batch[n], question_batch[n]],
        y=y_batch[n])
print(h.params)

len(vectors)
# On average there were 3.366419707987534 answer words and 105.92495348120413 other words per row.
# Positive should be weighted about 31.465165567405347.

# df = pickle_me(df, "C:/Users/Joseph Giroux/Datasets/df.pkl")