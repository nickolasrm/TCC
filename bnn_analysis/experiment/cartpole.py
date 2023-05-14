"""Gym cartpole solved with a genetic algorithm."""
from bnn_analysis.experiment.reinforcement import ga_experiment

cartpole = ga_experiment("cartpole", "CartPole-v1")

if __name__ == "__main__":
    cartpole()  # pylint: disable=no-value-for-parameter
