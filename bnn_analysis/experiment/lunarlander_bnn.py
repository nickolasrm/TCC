"""Gym lunarlander solved with a genetic algorithm and bit neural networks."""
from bnn_analysis.experiment.reinforcement import ga_experiment

lunarlander = ga_experiment("lunarlander_bnn", "LunarLander-v2")

if __name__ == "__main__":
    lunarlander()  # pylint: disable=no-value-for-parameter
