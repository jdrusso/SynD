from synd.core import load_model
from examples.data.simple_model import initial_distribution

n_steps = 10

if __name__ == '__main__':

    synmd_model = load_model("simple_synmd_model.dat")

    trajectory = synmd_model.generate_trajectory(
        initial_distribution=initial_distribution,
        n_steps=n_steps
    )

    print(trajectory)
