"""Gym cartpole solved with a genetic algorithm and bit neural networks."""
from bnn_analysis.experiment.reinforcement import ga_experiment

cartpole = ga_experiment("cartpole_bnn", "CartPole-v1")

if __name__ == "__main__":
    cartpole()  # pylint: disable=no-value-for-parameter
