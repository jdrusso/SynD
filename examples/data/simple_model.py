import numpy as np

transition_matrix = np.array([
    [0.75, 0.2, 0.05],
    [0.25, 0.6, 0.15],
    [0.05, 0.1, 0.85],
])

state_definitions = np.array([
    [1, 10, 100, 1000],
    [3, 30, 300, 3000],
    [5, 50, 500, 5000]
])

initial_distribution = np.array([1, 1, 0, 1, 2])

n_steps = 10


def backmapper(index: int) -> np.array:
    return state_definitions[index]
