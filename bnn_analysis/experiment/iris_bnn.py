"""Train a binarized neural network on the MinMax Scaled Iris dataset."""
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from wandb.keras import WandbMetricsLogger  # pylint: disable=no-name-in-module

from bnn_analysis.base.experiment import Metrics, experiment
from bnn_analysis.base.supervised import train
from bnn_analysis.experiment.iris import IrisDataSet


class ScaledIrisDataSet(IrisDataSet):
    """Get Iris data and splits it into train and test data."""

    def __init__(self):
        """Scales the original dataset using the train data."""
        super().__init__()
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)


@experiment
def iris_bnn(cfg: DictConfig) -> Metrics:
    """Train a BNN model on the MinMax Scaled Iris dataset."""
    layers = cfg.layers
    compile = cfg.compile  # pylint: disable=redefined-builtin
    fit = cfg.fit
    fit["callbacks"] = [WandbMetricsLogger()]
    return train(ScaledIrisDataSet(), layers, compile, fit, "classification")


if __name__ == "__main__":
    iris_bnn()  # pylint: disable=no-value-for-parameter
