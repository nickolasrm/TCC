"""Gym lunarlander solved with a genetic algorithm."""
from bnn_analysis.experiment.reinforcement import ga_experiment

lunarlander = ga_experiment("lunarlander", "LunarLander-v2")

if __name__ == "__main__":
    lunarlander()  # pylint: disable=no-value-for-parameter
