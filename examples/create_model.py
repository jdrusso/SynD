from synd.models.discrete.markov import MarkovGenerator
from examples.data.simple_model import transition_matrix, backmapper


if __name__ == '__main__':

    synmd_model = MarkovGenerator(
        transition_matrix=transition_matrix,
        backmapper=backmapper
    )

    synmd_model.save("simple_synmd_model.dat")
