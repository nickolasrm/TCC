"""Train a traditional neural network on the Boston Housing dataset."""
from sklearn.preprocessing import StandardScaler

from bnn_analysis.base.supervised import KerasDataSet
from bnn_analysis.experiment.supervised import supervised_experiment


class BostonHousingDataSet(KerasDataSet):
    """Get Boston Housing data and splits it into train and test data."""

    def __init__(self):
        """Prepare the dataset."""
        super().__init__("boston_housing")


class StdBostonHousingDataSet(BostonHousingDataSet):
    """Boston Housing data scaled to [0, 1]."""

    def __init__(self):
        """Prepare the dataset."""
        super().__init__()
        scaler = StandardScaler().fit(self.x_train)
        self.map_x(scaler.transform)


housing = supervised_experiment("housing", StdBostonHousingDataSet(), "regression")


if __name__ == "__main__":
    housing()  # pylint: disable=no-value-for-parameter
