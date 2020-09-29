import numpy as np


class NeuralNetwork:
    ACTIVATIONS = {
        "tanh": (lambda X: np.tanh(X), lambda X: 1.0 - np.square(X)),
        "sin": (lambda X: np.sin(X), lambda X: -np.cos(X)),
        "relu": (lambda X: np.maximum(X, 0, X), lambda X: np.greater(X, 0).astype(int)),
        "sigmoid": (lambda X: 1.0 / (1.0 + np.exp(-X)), lambda X: X * (1.0 - X)),
        "linear": (
            lambda X: np.array([-1 if x < -1 else 1 if x > 1 else x for x in X]),
            lambda X: np.array([0 if x < -1 or x > 1 else 1 for x in X]),
        ),
    }
    ERROR = {
        "mean_squared_error": lambda y, yhat: y - yhat,
        "cross_entropy": lambda y, yhat: (y - yhat) / ((1 - yhat) * yhat),
    }

    def __init__(
        self,
        shape,
        activation="tanh",
        output_activation="sigmoid",
        learning_rate=0.1,
        loss="mean_squared_error",
    ):
        self.__prev_error = None
        self.__crt_error = None

        self.__weights = []
        self.__activations = []
        self.__biases = []

        self.__shape = shape
        self.__learning_rate = learning_rate

        self.__loss = loss
        self.__error_function = self.ERROR.get(loss)

        self.__transfer, self.__transfer_derivative = self.ACTIVATIONS[activation]

        self.__output_transfer, self.__output_transfer_derivative = self.ACTIVATIONS[
            output_activation
        ]

        self.__activations = [np.zeros(layer_length) for layer_length in self.__shape]
        self.__biases = [
            np.random.randn(layer_length) for layer_length in self.__shape[1:]
        ]
        self.__weights = [
            np.random.randn(self.__shape[i - 1], layer_length)
            for i, layer_length in enumerate(self.__shape)
            if i > 0
        ]

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, new_weights):
        self.__weights = new_weights

    @property
    def biases(self):
        return self.__biases

    @biases.setter
    def biases(self, new_biases):
        self.__biases = new_biases

    @property
    def learning_rate(self):
        return self.__learning_rate

    # this attribute is made public in order
    # to dynamically adapt the networks learning rate
    # as it approaches a viable solution
    @learning_rate.setter
    def learning_rate(self, new_rate):
        self.__learning_rate = new_rate

    def forward_pass(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if len(X) != self.__shape[0]:
            raise ValueError(
                "Input shape incorrect. Got len %s expected len %s"
                % (len(X), self.__shape[0])
            )

        self.__activations[0] = X

        for i in range(0, len(self.__activations) - 1):

            z = np.dot(self.__activations[i], self.__weights[i]) + self.__biases[i]

            if i == len(self.__activations) - 2:
                self.__activations[i + 1] = self.__output_transfer(z)

            else:
                self.__activations[i + 1] = self.__transfer(z)

        if self.__loss == "cross_entropy":
            return self.softmax(self.__activations[-1])

        return self.__activations[-1]

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backward_pass(self, network_input, network_output, expected_output):
        if not isinstance(expected_output, np.ndarray):
            expected_output = np.array(expected_output)

        if not isinstance(network_input, np.ndarray):
            network_input = np.array(network_input)

        if len(expected_output) != self.__shape[-1]:
            raise ValueError(
                "Expected output shape incorrect. Got len %s expected len %s"
                % (len(expected_output), len(self.__shape[-1]))
            )
        if len(network_input) != self.__shape[0]:
            raise ValueError(
                "Input shape incorrect. Got len %s expected len %s"
                % (len(network_input), len(self.__shape[0]))
            )

        error = self.__error_function(expected_output, network_output)

        # if not self.__prev_error:
        #     self.__prev_error = error
        # if not self.__crt_error:
        #     self.__crt_error = error

        # if 1 < self.__prev_error / self.__crt_error < 2:
        #     print("Prev error: ", self.__prev_error)
        #     print("Current error: ", self.__crt_error)
        #     print("Previous learning rate: ", self.learning_rate)
        #     print("New learning rate: "), self.learning_rate / 2
        #     self.learning_rate = self.learning_rate / 2

        delta = error * self.__output_transfer_derivative(network_output)

        for i in reversed(range(0, len(self.__weights))):

            # store these in a temporary varaible to such that the
            # error/delta of the next layer are calculated using the
            # the original weights, not the updated ones
            bias_update = delta * self.__learning_rate

            # TODO: figure out how to avoid these reshaping operations
            a = self.__activations[i].reshape(1, len(self.__activations[i]))
            d = delta.reshape(1, len(delta))

            weight_update = np.dot(a.T, d) * self.__learning_rate
            weight_update = weight_update.reshape(self.__weights[i].shape)

            # error / delta for the next layer of weights and biases
            error = np.dot(delta, self.__weights[i].T)
            delta = error * self.__transfer_derivative(self.__activations[i])

            self.__biases[i] += bias_update
            self.__weights[i] += weight_update

        # self.__prev_error = self.__crt_error
        # self.__crt_error = None
