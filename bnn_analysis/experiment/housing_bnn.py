"""Train a traditional neural network on the Boston Housing dataset."""
from sklearn.preprocessing import MinMaxScaler

from bnn_analysis.experiment.housing import StdBostonHousingDataSet
from bnn_analysis.experiment.supervised import supervised_experiment


housing_bnn = supervised_experiment(
    "housing_bnn", StdBostonHousingDataSet(), "regression"
)


if __name__ == "__main__":
    housing_bnn()  # pylint: disable=no-value-for-parameter
