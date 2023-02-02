"""Base class for training Gym reinforcement environments."""
import typing as t

import gym
import numpy as np
from keras import Model, Sequential
from keras.layers import Input, Layer
from pygad import GA, kerasga

from bnn_analysis.utils import get_logger

Fitness = t.Callable[[np.ndarray, int], float]


class KerasGymGA:
    """Simplifies the genetic optimization with Keras models on Gym environments."""

    def __init__(
        self,
        env_id: str,
    ):
        """Initialize the class.

        Args:
            env_id: Gym environment ID.

        """
        self.env_id = env_id
        self._model = None

    def build(self, *layers: Layer, **kwargs):
        """Build a Keras model from a list of layers.

        This method will automatically add the input layer and compile the model.

        Args:
            layers: List of Keras layers.
            **kwargs: Additional arguments to pass to the model's compile method.

        """
        # Getting the input and output sizes from the environment
        env = self.env
        input_size = env.observation_space.shape[0]  # type: ignore
        output_size = env.action_space.n  # type: ignore

        # Adding the input layer
        model = Sequential()
        model.add(Input(input_size))

        # Checking if the last layer is the output layer
        if layers[-1].units != output_size:
            raise ValueError(
                f"Environment expects {output_size} outputs, "
                f"but the last layer has {layers[-1].units}"
            )

        # Adding layers
        for layer in layers:
            model.add(layer)
        model.compile(**kwargs)

        self._model = model

    def fit(self, population_size: int = 10, **kwargs):
        """Train a Keras model using a genetic algorithm.

        Args:
            population_size: Defaults to 10.
            **kwargs: Additional arguments to pass to the genetic algorithm.

        """
        env = self.env
        model = self.model
        genetic = GA(
            **{
                "num_generations": 50,
                "num_parents_mating": 5,
                "on_generation": _callback_generation,
                **kwargs,
            },
            initial_population=kerasga.KerasGA(
                model, population_size
            ).population_weights,
            fitness_func=self._pygad_fitness(),
            stop_criteria=f"reach_{env.spec.reward_threshold}",
        )
        genetic.run()
        _replace_weights(model, genetic.best_solution())

    def _pygad_fitness(self) -> Fitness:
        # Generate a fitness function for a given environment and model.

        def fitness(solution: np.ndarray, _: t.Any) -> float:
            model = self.model
            _replace_weights(model, solution)

            env = self.env
            observation, _ = env.reset()
            reward_accumulator = 0.0
            finished = False
            while not finished:
                action = _predict(self.model, observation)
                observation, reward, done, truncated, _ = env.step(action)
                reward_accumulator += reward
                finished = done or truncated
            return reward_accumulator

        return fitness

    @property
    def env(self) -> gym.Env:
        """Get the Gym environment."""
        return gym.make(self.env_id)

    @property
    def model(self) -> Model:
        """Get the model if it has been built."""
        if self._model is None:
            raise ValueError("Model not built")
        return self._model


def _predict(model: Model, data: np.ndarray) -> int:
    predictions = model.predict(data.reshape(1, -1), verbose="0")
    return np.argmax(predictions)  # type: ignore


def _replace_weights(model: Model, solution: np.ndarray):
    # Replace the weights of the model with the weights of the solution.
    solution_weights = kerasga.model_weights_as_matrix(
        model=model, weights_vector=solution
    )
    model.set_weights(solution_weights)


def _callback_generation(genetic: GA):
    get_logger().info(
        {
            "generation": genetic.generations_completed,
            "fitness": genetic.best_solution()[1],
        }
    )
