import pickle

def pickle_me(obj, filename):
    with open(filename, 'wb') as fh:
        pickle.dump(obj, fh)

def unpickle_me(filename):
    with open(filename, 'rb') as fh:
        return pickle.load(fh)

def add_dim(mat):
    return mat.reshape([1]+list(mat.shape))
