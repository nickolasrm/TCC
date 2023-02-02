"""Dense layers with binarized weights and activations."""
import tensorflow as tf
from keras.layers import Dense

from bnn_analysis.base import constraints


class BinarizedDense(Dense):  # pylint: disable=too-few-public-methods
    """A dense layer with binarized weights and activations.

    This layer converts all weights to either -1 or 1 and all activations to
    either -1 or 1.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        units: int,
        activation: str = "softsign",
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        **kwargs
    ):
        """Initialize the layer.

        Args:
            units: Number of neurons
            activation: Regular activation, should be similar to the binarizer function
                in order to use the straight through estimator. Defaults to "softsign".
            use_bias: Whether to add bias weights. Defaults to True.
            kernel_initializer: Sets the initial values for the kernel.
                Defaults to "glorot_uniform".
            bias_initializer: Sets the initial values for the bias. Defaults to "zeros".

        """
        super().__init__(
            units,
            activation,
            use_bias,
            kernel_initializer,
            bias_initializer,
            None,
            None,
            None,
            None,
            None,
            **kwargs
        )
        self.constraint = constraints.get("sign")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Binarized feed forward."""
        outputs = tf.matmul(a=inputs, b=self.constraint(self.kernel))
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.constraint(self.bias))
        outputs = self.activation(outputs)
        return self.constraint(outputs)
