# pylint: disable=arguments-differ
"""Benchmark callable objects."""
from __future__ import annotations

import pprint
import timeit
import typing as t
from dataclasses import dataclass, field

import numpy as np
import tqdm
from pympler import asizeof

from bnn_analysis.utils import prefix_dict


@dataclass
class Measure:
    """Measure a quantity."""

    name: str
    unit: t.Optional[str] = None
    observations: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64), init=False
    )

    def add(self, observation: float) -> None:
        """Add an observation to the measure."""
        self.observations = np.append(self.observations, observation)

    @property
    def mean(self) -> float:
        """Return the mean of the observations."""
        return np.mean(self.observations)

    @property
    def std(self) -> float:
        """Return the standard deviation of the observations."""
        return np.std(self.observations)

    @property
    def min(self) -> float:
        """Return the minimum of the observations."""
        return np.min(self.observations)

    @property
    def max(self) -> float:
        """Return the maximum of the observations."""
        return np.max(self.observations)

    @property
    def median(self) -> float:
        """Return the median of the observations."""
        return np.median(self.observations)

    @property
    def empty(self) -> bool:
        """Return True if the measure has no observations."""
        return len(self.observations) == 0

    def report(self) -> t.Dict[str, t.Any]:
        """Return a report of the measure."""
        if len(self.observations) > 1:
            report: t.Dict[str, t.Any] = {
                "min": self.min,
                "max": self.max,
                "median": self.median,
                "mean": self.mean,
                "std": self.std,
            }
        else:
            report = {"value": self.mean}
        if self.unit is not None:
            report["unit"] = self.unit
        return prefix_dict(report, self.name)

    def measure(self, *args, **kwargs):
        """Measure the quantity."""
        raise NotImplementedError("Measure.measure is not implemented.")


@dataclass
class MemoryMeasure(Measure):
    """Measure the memory usage."""

    unit: str = field(default="kilobytes", init=False)

    def add(self, observation: float) -> None:
        """Add an observation to the measure."""
        super().add(observation / 1024)

    def measure(self, obj: t.Any):
        """Measure the memory usage of the object."""
        size = asizeof.asizeof(obj)
        self.add(size)


@dataclass
class TimeMeasure(Measure):
    """Measure the time to run a function."""

    unit: str = field(default="seconds", init=False)

    def measure(self, function: t.Callable[[], None]):
        """Measure the time to run the function on the input."""
        self.add(timeit.timeit(function, number=1))


@dataclass
class Benchmark:
    """Benchmark callable objects."""

    object: t.Callable
    input: t.Any
    repeat: int = 1000
    time: TimeMeasure = field(init=False)
    memory: MemoryMeasure = field(init=False)

    def __post_init__(self):
        """Initialize the benchmark."""
        self.time = TimeMeasure("time")
        self.memory = MemoryMeasure("memory")

    def measure_memory(self):
        """Measure the memory usage of the object."""
        self.memory.measure(self.object)

    def loop(self, function: t.Callable[[], None]):
        """Run the function `repeat` times."""
        for _ in tqdm.tqdm(range(self.repeat)):
            function()

    def measure_time(self):
        """Measure the time to run the object on the input."""
        self.loop(lambda: self.time.measure(lambda: self.object(self.input)))

    def report(self) -> t.Dict[str, t.Any]:
        """Return a report of the benchmark."""
        reports = [
            attribute.report()
            for attribute in self.__dict__.values()
            if isinstance(attribute, Measure) and not attribute.empty
        ]
        report = {key: value for report in reports for key, value in report.items()}
        return {
            **report,
            "benchmark.repeat": self.repeat,
        }

    def measure(self):
        """Call the measure methods."""
        self.measure_memory()
        self.measure_time()

    def run(self) -> None:
        """Run the benchmark."""
        self.measure()
        pprint.pprint(self.report())

    def compare(self, other: Benchmark) -> t.Dict[str, t.Any]:
        """Compare the benchmark with another benchmark."""
        return {
            key: value / other.report()[key] if isinstance(value, float) else value
            for key, value in self.report().items()
        }
