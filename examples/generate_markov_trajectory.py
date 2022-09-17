from synd.models.discrete.markov import MarkovGenerator
from examples.data.simple_model import transition_matrix, initial_distribution, n_steps, backmapper


if __name__ == '__main__':

    synmd_model = MarkovGenerator(
        transition_matrix=transition_matrix,
        backmapper=backmapper
    )

    trajectory = synmd_model.generate_trajectory(
        initial_distribution=initial_distribution,
        n_steps=n_steps
    )

    print(trajectory)
