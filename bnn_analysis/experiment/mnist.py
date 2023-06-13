"""Run a supervised experiment on the MNIST dataset."""
from keras.utils import to_categorical

from bnn_analysis.base.supervised import KerasDataSet
from bnn_analysis.experiment.supervised import supervised_experiment


class MNISTDataSet(KerasDataSet):
    """Get MNIST data and splits it into train and test data."""

    def __init__(self):
        """Prepare the dataset."""
        super().__init__(name="mnist")
        self.map_y(lambda y: to_categorical(y, num_classes=10))


mnist = supervised_experiment("mnist", MNISTDataSet(), "classification")

if __name__ == "__main__":
    mnist()  # pylint: disable=no-value-for-parameter
