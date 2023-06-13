"""Train a binarized neural network on the MinMax Scaled Iris dataset."""
from sklearn.preprocessing import MinMaxScaler

from bnn_analysis.experiment.iris import IrisDataSet
from bnn_analysis.experiment.supervised import supervised_experiment


class ScaledIrisDataSet(IrisDataSet):
    """Get Iris data and splits it into train and test data."""

    def __init__(self):
        """Scales the original dataset using the train data."""
        super().__init__()
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.x_train)
        self.map_x(scaler.transform)


iris_bnn = supervised_experiment("iris_bnn", ScaledIrisDataSet(), "classification")


if __name__ == "__main__":
    iris_bnn()  # pylint: disable=no-value-for-parameter
