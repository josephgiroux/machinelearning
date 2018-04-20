import pickle

def pickle_me(obj, filename):
    with open(filename, 'wb') as fh:
        pickle.dump(obj, fh)

def unpickle_me(filename):
    with open(filename, 'r') as fh:
        return pickle.load(fh.read())