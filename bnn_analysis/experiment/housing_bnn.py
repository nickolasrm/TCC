"""Train a traditional neural network on the Boston Housing dataset."""
from sklearn.preprocessing import MinMaxScaler

from bnn_analysis.experiment.housing import BostonHousingDataSet
from bnn_analysis.experiment.supervised import supervised_experiment


class ScaledBostonHousingDataSet(BostonHousingDataSet):
    """Boston Housing data scaled to [-1, 1]."""

    def __init__(self):
        """Prepare the dataset."""
        super().__init__()
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.x_train)
        self.map_x(scaler.transform)


housing_bnn = supervised_experiment(
    "housing_bnn", ScaledBostonHousingDataSet(), "regression"
)


if __name__ == "__main__":
    housing_bnn()  # pylint: disable=no-value-for-parameter
