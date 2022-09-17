import pickle


def load_model(filename: str):

    with open(filename, 'rb') as infile:
        model = pickle.load(infile)

    return model
