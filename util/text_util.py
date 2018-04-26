from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

WORD2VEC_PATH = 'C:/Users/Joseph Giroux/Datasets/GoogleNews-vectors-negative300.bin'

def get_word2vec_model(filename=WORD2VEC_PATH):
    model = KeyedVectors.load_word2vec_format(
        filename, binary=True)
    return model

w2v = get_word2vec_model()
model.most_similar_cosmul(positive=['Baghdad', 'England'], negative=['London'])

model.most_similar(positive=['Baghdad', 'England'], negative=['London'])

model.most_similar_cosmul(positive=['pilot', 'ship'], negative=['jet'])

model.most_similar_cosmul(positive=['writer', 'sing'], negative=['write'])

# get a vector
# model.get_vector('word')



