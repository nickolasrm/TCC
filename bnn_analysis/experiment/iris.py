"""Train a traditional neural network on the Iris dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

from bnn_analysis.base.supervised import HuggingFaceDataSet
from bnn_analysis.experiment.supervised import supervised_experiment


class IrisDataSet(HuggingFaceDataSet):
    """Get Iris data and splits it into train and test data."""

    def __init__(self):
        """Prepare the dataset."""
        target = "Species"
        dataset = self.download("scikit-learn/iris")["train"].to_pandas()
        data, labels = dataset.drop(target, axis=1), pd.get_dummies(dataset[target])
        x_train, x_test, y_train, y_test = (
            data.to_numpy() for data in train_test_split(data, labels, test_size=0.2)
        )
        super().__init__(x_train, y_train, x_test, y_test)


iris = supervised_experiment("iris", IrisDataSet(), "classification")

if __name__ == "__main__":
    iris()  # pylint: disable=no-value-for-parameter
