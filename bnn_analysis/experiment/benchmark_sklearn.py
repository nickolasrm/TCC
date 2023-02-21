"""Benchmark Sklearn MLPs."""
import pandas as pd
from omegaconf import DictConfig

from bnn_analysis.base.experiment import experiment
from bnn_analysis.base.mlp import SklearnMLP
from bnn_analysis.experiment.benchmark import MLPBenchmark


@experiment
def benchmark_sklearn(cfg: DictConfig):
    """Compare the performance of a regular neural network with a BNN."""
    report = {}
    for name, spec in cfg.cases.items():
        mlp = SklearnMLP(spec.inputs)
        for unit in spec.units:
            mlp.add(unit)
        mlp.compile()
        bench = MLPBenchmark(mlp, mlp.generate_inputs(), 2000)
        bench.run()
        report[name] = bench.report()

    df = pd.DataFrame([{"name": name, **metrics} for name, metrics in report.items()])

    return df


if __name__ == "__main__":
    benchmark_sklearn()  # pylint: disable=no-value-for-parameter
