"""Benchmark BNN and Keras MLPs."""
import gc
import typing as t
from dataclasses import dataclass, field

import pandas as pd
from omegaconf import DictConfig

from bnn_analysis.base.benchmark import Benchmark, MemoryMeasure
from bnn_analysis.base.bit_mlp import BitMLP
from bnn_analysis.base.experiment import experiment
from bnn_analysis.base.mlp import MLP, KerasMLP


class MLPBenchmark(Benchmark):
    """Benchmark a network."""

    parameters_memory: MemoryMeasure = field(init=False)

    def __post_init__(self):
        """Initialize the benchmark."""
        super().__post_init__()
        self.parameters_memory = MemoryMeasure("parameters_memory")

    @property
    def network(self) -> MLP:
        """Return the network."""
        return t.cast(MLP, self.object)

    def measure_parameters_memory(self):
        """Measure the memory usage of the parameters."""
        self.parameters_memory.measure(self.network.get_weights())

    def measure(self):
        """Call the measure methods."""
        self.measure_parameters_memory()
        super().measure()


@dataclass
class Case:
    """A single case of the benchmark."""

    inputs: int
    units: t.List[int]
    bit_mlp: MLPBenchmark = field(init=False)
    keras_mlp: MLPBenchmark = field(init=False)

    def build(self, cls_type: t.Type[MLP]) -> MLP:
        """Build a network."""
        mlp = cls_type(self.inputs)
        for unit in self.units:
            mlp.add(unit)
        mlp.compile()
        return mlp

    @classmethod
    def benchmark(cls, mlp: MLP) -> MLPBenchmark:
        """Run the benchmark."""
        bench = MLPBenchmark(mlp, mlp.generate_inputs(), 2000)
        bench.run()
        gc.collect()
        return bench

    def run(self):
        """Run the benchmark."""
        bit_mlp = self.build(BitMLP)
        keras_mlp = self.build(KerasMLP)
        self.bit_mlp = self.benchmark(bit_mlp)
        self.keras_mlp = self.benchmark(keras_mlp)

    def report(self):
        """Return the reports."""
        bit_report = self.bit_mlp.report()
        keras_report = self.keras_mlp.report()
        comparison = self.keras_mlp.compare(self.bit_mlp)
        return {
            "bit": bit_report,
            "keras": keras_report,
            "comparison": comparison,
        }


@experiment
def benchmark(cfg: DictConfig):
    """Compare the performance of a regular neural network with a BNN."""
    report = {}
    for name, spec in cfg.cases.items():
        case = Case(spec.inputs, spec.units)
        case.run()
        report[name] = case.report()

    df = pd.DataFrame(
        [
            {"name": name, "case": report, **metrics}
            for name, reports in report.items()
            for report, metrics in reports.items()
        ]
    )

    return df


if __name__ == "__main__":
    benchmark()  # pylint: disable=no-value-for-parameter
