"""Base class for training Gym reinforcement environments."""
import typing as t
from dataclasses import dataclass, field

import gym
import numpy as np
import torch
from pygad import GA, torchga
from torch import nn

from bnn_analysis.base.experiment import Metrics
from bnn_analysis.base.trainer import TorchTrainer
from bnn_analysis.utils import get_logger

Fitness = t.Callable[[np.ndarray, int], float]


@dataclass
class ReinforcementTrainer(TorchTrainer):
    """Base class for reinforcement learning training."""


@dataclass
class GATrainer(ReinforcementTrainer):
    """Genetic algorithm trainer for reinforcement learning."""

    fitness: Fitness
    report: t.Dict[str, t.Any] = field(default_factory=dict, init=False)

    def _fit(self, population_size: t.Optional[int] = None, **kwargs: t.Any):
        population_size = population_size or 10
        hyperparameters = {
            **dict(
                num_generations=10,
                num_parents_mating=10,
                mutation_probability=0.1,
            ),
            **kwargs,
        }
        population = torchga.TorchGA(
            model=self.model,
            num_solutions=population_size,
        )
        genetic = GA(
            fitness_func=self.fitness,
            initial_population=population.population_weights,
            keep_parents=-1,
            on_generation=lambda x: self._callback_generation(  # pylint: disable=W0108
                x
            ),
            **hyperparameters,
        )
        genetic.run()
        self._replace_weights(genetic.best_solution()[0])

    def _replace_weights(self, solution: np.ndarray):
        # Replace the weights of the model with the weights of the solution.
        solution_weights = torchga.model_weights_as_dict(
            model=self.model, weights_vector=solution
        )
        self.model.load_state_dict(solution_weights)

    def _callback_generation(self, genetic: GA):
        iteration = genetic.generations_completed
        best_score = genetic.best_solution()[1]
        mean_score = np.mean(genetic.last_generation_fitness)
        std_score = np.std(genetic.last_generation_fitness)
        get_logger().info(
            {
                "generation": iteration,
                "best_fitness": best_score,
                "mean_fitness": mean_score,
                "std_fitness": std_score,
            }
        )
        self.report.setdefault("best_fitness", []).append(best_score)
        self.report.setdefault("avg_fitness", []).append(mean_score)
        self.report.setdefault("std_fitness", []).append(std_score)

    def evaluate(self) -> Metrics:
        """Evaluate the model on the Gym environment."""
        return self.report


@dataclass
class GymTrainer(GATrainer):
    """Trainer for Gym environments."""

    fitness: Fitness = field(init=False)
    env_id: str

    def __post_init__(self):
        """Initialize the Gym environment."""
        self.fitness = self._build_fitness()

    @property
    def input_shape(self) -> t.Tuple[int, ...]:
        """Get the Gym environment input shape for the neural network."""
        shape = self.env.observation_space.shape
        if shape is None:
            raise ValueError("Gym environment has no observation space.")
        return shape

    @property
    def output_shape(self) -> t.Tuple[int, ...]:
        """Get the Gym environment output shape for the neural network."""
        outputs = self.env.action_space.n  # type: ignore
        return (outputs,)

    @property
    def env(self) -> gym.Env:
        """Get the Gym environment."""
        return gym.make(self.env_id)

    def _build_fitness(self) -> Fitness:
        # Generate a fitness function for a given environment and model.

        def fitness(solution: np.ndarray, _: t.Any) -> float:
            self._replace_weights(solution)

            env = self.env
            observation, _ = env.reset()
            reward_accumulator = 0.0
            finished = False
            while not finished:
                action = self._predict(observation)
                observation, reward, done, truncated, _ = env.step(action)
                reward_accumulator += reward
                finished = done or truncated
            return reward_accumulator

        return fitness

    def _predict(self, data: np.ndarray) -> int:
        tensor = torch.tensor(data, dtype=torch.float)
        predictions = self.model(tensor)
        action = int(predictions.argmax())
        return action


def train(
    layers: t.List[nn.Module],
    fit: t.Optional[dict] = None,
    solver: t.Optional[t.Literal["gym"]] = None,
    init: t.Optional[dict] = None,
) -> Metrics:
    """Trains a model using reinforcement learning.

    Args:
        layers: PyTorch layers of the model.
        fit: Parameters for fitting the model.
        solver: Name of the solver to use.
        init: Parameters for initializing the trainer.

    Returns:
        Training execution metrics

    """
    init, fit, solver = init or {}, fit or {}, solver or "gym"
    if solver != "gym":
        raise ValueError(f"Unknown solver {solver}")
    cls = GymTrainer
    trainer = cls(**init)
    trainer.build(*layers)
    trainer.fit(**fit)
    report = trainer.evaluate()
    return report
