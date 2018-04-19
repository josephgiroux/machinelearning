from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors


def get_word2vec_model(filename='C:/Users/Joseph Giroux/Datasets/GoogleNews-vectors-negative300.bin'):
    model = KeyedVectors.load_word2vec_format(
        filename, binary=True)
    return model

model = get_word2vec_model()
model.most_similar_cosmul(positive=['Baghdad', 'England'], negative=['London'])

model.most_similar(positive=['Baghdad', 'England'], negative=['London'])

model.most_similar_cosmul(positive=['pilot', 'ship'], negative=['jet'])

model.most_similar_cosmul(positive=['writer', 'sing'], negative=['write'])

# get a vector
# model.get_vector('word')



