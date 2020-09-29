import numpy as np


class NeuralNetwork:
    def __init__(
        self,
        shape: list,
        activation: str = "tanh",
        output_activation: str = "sigmoid",
        learning_rate: float = 0.02,
        loss="mean_squared_error",
    ):

        # Publicly accessible attribute in order to be able to
        # implement dynamically updated learning rate later
        self.__learning_rate = learning_rate

        # create layers of the neural net
        self.__layers = []

        for layer_index, layer_length in enumerate(shape):
            # output layer is initialised with linear activation function

            neuron_params = {
                "activation_function": activation,
                "learning_rate": learning_rate,
                "loss": loss,
            }

            if layer_index == len(shape) - 1:
                neuron_params["activation_function"] = output_activation

            self.__layers.append([Neuron(**neuron_params) for _ in range(layer_length)])

        # fully connects each layer to the next
        for layer_index, layer in enumerate(self.__layers):

            # output layer has already been connected to by the previous
            # layer, and has no other layer to connect to
            if layer_index == len(self.__layers) - 1:
                break

            for neuron in layer:
                # connect neuron to each neuron of the next layer
                for target_neuron in self.__layers[layer_index + 1]:

                    synapse = Synapse(neuron_in=neuron, neuron_out=target_neuron)

                    neuron.synapses_out.append(synapse)
                    target_neuron.synapses_in.append(synapse)

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        for layer in self.__layers:
            for neuron in layer:
                neuron.learning_rate = learning_rate

    @property
    def output_layer(self):
        return self.__layers[-1]

    def forward_pass(self, network_input):
        if len(network_input) != len(self.__layers[0]):
            raise ValueError("Input not the same shape as input layer")

        for layer_index, layer in enumerate(self.__layers):
            for neuron_index, neuron in enumerate(layer):

                # activate the first layer neurons with a single
                # input as opposed as the previous layer's outputs
                if layer_index == 0:
                    neuron.activate(network_input[neuron_index])
                else:
                    neuron.activate()

        return np.array([neuron.output for neuron in self.output_layer])

    def backward_pass(self, network_input, network_output, expected_output):
        if len(expected_output) != len(self.__layers[-1]):
            raise ValueError("Expected output not the same shape as actual output")

        for layer_index, layer in reversed(list(enumerate(self.__layers))):
            for neuron_index, neuron in enumerate(layer):

                # calculate output layer error based on the expected value
                # as oppsed to the weight/error from the next connected layer
                if layer_index == len(self.__layers) - 1:
                    neuron.calculate_error(expected_output[neuron_index])
                else:
                    neuron.calculate_error()

                neuron.update_weights()


class Neuron:

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
        self, learning_rate=0.02, activation_function="tanh", loss="mean_squared_error"
    ):

        self.__synapses_in = []
        self.__synapses_out = []
        self.__bias = np.random.uniform(-1, 1)
        self.__learning_rate = learning_rate
        self.__transfer, self.__transfer_derivative = self.ACTIVATIONS[
            activation_function
        ]
        self.__error_function = self.ERROR.get(loss)

    @property
    def transfer(self):
        return self.__transfer

    @transfer.setter
    def transfer(self, activation_function):
        self.__transfer, self.__transfer_derivative = self.ACTIVATIONS[
            activation_function
        ]

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, value):
        self.__bias = value

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

    @property
    def synapses_in(self):
        return self.__synapses_in

    @synapses_in.setter
    def synapses_in(self, synapses):
        self.__synapses_in = synapses

    @property
    def synapses_out(self):
        return self.__synapses_out

    @synapses_out.setter
    def synapses_out(self, synapses):
        self.__synapses_out = synapses

    def calculate_error(self, expected=None):

        if expected:
            self.__error = self.__error_function(expected, self.output)
        else:

            self.__error = np.sum(
                [
                    synapse.neuron_out.delta * synapse.weight
                    for synapse in self.synapses_out
                ]
            )

        self.delta = self.__error * self.__transfer_derivative(self.output)

    def update_weights(self):
        for synapse in self.synapses_in:

            synapse.weight += (
                self.__learning_rate * self.delta * synapse.neuron_in.output
            )

        self.__bias += self.delta * self.__learning_rate

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, output):
        self.__output = output

    def activate(self, network_input=None):

        if network_input is None:

            inputs = [synapse.neuron_in.output for synapse in self.__synapses_in]
            weights_in = [synapse.weight for synapse in self.__synapses_in]

            self.__activation = np.dot(weights_in, inputs) + self.__bias

            self.output = self.__transfer(self.__activation)

        # input was directly provided to neuron, meaning this is a
        # first layer neuron. In this case, each Neuron has only one
        # input and one weight. This will be used to calculate its transfer
        # function
        else:
            if isinstance(network_input, list) and len(network_input) != 1:
                raise ValueError("Neuron input should be a single element")

            if isinstance(network_input, list):
                network_input = network_input[0]

            self.output = network_input


class Synapse:
    def __init__(self, neuron_in, neuron_out):
        self.__neuron_in = neuron_in
        self.__neuron_out = neuron_out
        self.__weight = np.random.uniform(-1, 1)

    @property
    def neuron_in(self):
        return self.__neuron_in

    @property
    def neuron_out(self):
        return self.__neuron_out

    @property
    def weight(self):
        return self.__weight

    @weight.setter
    def weight(self, value):
        self.__weight = value
