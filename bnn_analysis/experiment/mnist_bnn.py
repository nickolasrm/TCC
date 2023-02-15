"""Run a supervised experiment on the MNIST dataset with BNNs."""
import numpy as np

from bnn_analysis.experiment.mnist import MNISTDataSet
from bnn_analysis.experiment.supervised import supervised_experiment


class DiscretizedMNISTDataSet(MNISTDataSet):
    """A MNIST dataset discretized into -1 and 1."""

    def __init__(self):
        """Initialize the dataset."""
        super().__init__()
        ref_data = self.x_train
        ref_data = ref_data[ref_data > 0]  # remove black pixels
        ref_data = np.mean(ref_data)
        self.map_x(lambda x: np.where(x > ref_data, 1, -1))


mnist_bnn = supervised_experiment(
    "mnist_bnn", DiscretizedMNISTDataSet(), "classification"
)

if __name__ == "__main__":
    mnist_bnn()  # pylint: disable=no-value-for-parameter
