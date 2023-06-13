"""Benchmark the C++ implementation of the MLP."""
import subprocess
import typing as t
from dataclasses import dataclass

import pandas as pd
from omegaconf import DictConfig

from bnn_analysis import BASE_PATH, PACKAGE
from bnn_analysis.base.benchmark import Benchmark
from bnn_analysis.base.experiment import experiment


@dataclass
class CppNN:
    """A C++ implementation of the neural network."""

    binary: str

    def __call__(self, args: t.List[str]) -> float:
        """Run a binary with the given arguments and return the time spent."""
        parsed_args = [str(arg) for arg in args]
        process = subprocess.run(
            [(BASE_PATH / PACKAGE / "cpp" / self.binary).as_posix(), *parsed_args],
            check=True,
            capture_output=True,
        )
        return float(process.stdout.decode("utf-8").strip())


@dataclass
class CppBenchmark(Benchmark):
    """Benchmark the C++ implementation of the network."""

    object: CppNN

    def measure_time(self):
        """Measure the time spent by the benchmark."""
        self.loop(lambda: self.time.add(self.object(self.input)))

    def measure(self):
        """Measure the time spent by the benchmark."""
        self.measure_time()


def benchmark_binary(binary: str, case: DictConfig) -> CppBenchmark:
    """Benchmark a binary."""
    network = CppNN(binary)
    bench = CppBenchmark(network, [case.inputs, *case.units], 2000)
    bench.run()
    return bench


@experiment
def benchmark_cpp(cfg: DictConfig):
    """Benchmark the C++ implementation of the BNN."""
    report = dict()
    for name, spec in cfg.cases.items():
        bit_bench = benchmark_binary("bnn", spec)
        float_bench = benchmark_binary("nn", spec)
        report[name] = {
            "bit": bit_bench.report(),
            "float": float_bench.report(),
            "comparison": float_bench.compare(bit_bench),
        }

    df = pd.DataFrame(
        [
            {"name": name, "case": report, **metrics}
            for name, reports in report.items()
            for report, metrics in reports.items()
        ]
    )

    return df


if __name__ == "__main__":
    benchmark_cpp()  # pylint: disable=no-value-for-parameter
