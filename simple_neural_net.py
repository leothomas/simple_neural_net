import numpy as np

# TODO: Re-write, as an exercie, this class using purely matrix
# operations. This class is implemented in an OOP manner, which
# I assume will be brutal on CPU.
# In a matrix oriented class, each layer N is an array [ a0, a1,...,an ]
# the synapses from the previous later are a matrix W =
# [ [ w00, w01,...,w0n ],
#   [ w10, w11,...,w1n ],
#   ...
#   [wm0, wm1, ..., wmn]
# ]
# Where n is the number of neurons in layer N, and m
# is the number of neurons in the layer previous to N, which
# are connected to the neurons in layer N

# TODO: dynamically set activation function
# TODO: decrease learning rate proportionally to error


class Neuron:
    def __init__(self, learning_rate=0.1):

        self.__synapses_in = []
        self.__synapses_out = []
        self.__bias = np.random.uniform(-1, 1)
        self.__learning_rate = learning_rate

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

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1. - x * x

    def ReLU(self, x):
        return x * (x > 0)

    def dReLU(self, x):
        return 1. * (x > 0)

    def sigmoid(self, x):
        # basic sigmoid
        return 1.0 / (1.0 + np.exp(-x))

    def dsigmoid(self, x):
        return x * (1.0 - x)

    def linear(self, x):
        if x < -1:
            return -1
        if x > 1:
            return 1
        return x

    def dlinear(self, x):
        if x < -1 or x > 1:
            return 0
        return 1

    def transfer(self, x):
        if self.__synapses_out == []:
            return self.linear(x)
        return self.tanh(x)

    def transfer_derivative(self, x):
        if self.__synapses_out == []:
            return self.dlinear(x)
        return self.dtanh(x)

    def calculate_error(self, expected=None):

        if expected:
            # I'm unsure of this error, calculated on the output
            # neuron. Many sources calculate the errror as a square
            # of the difference  o the expected and actual values,
            # which makes sense to penalize large differences
            # It seems however that squaring this value also makes it
            # positive in all cases, which only ever increases the
            # weights and biases, which leads to completely inaccurate
            # results
            self.__error = expected - self.output
        else:

            self.__error = np.sum([
                synapse.neuron_out.delta * synapse.weight
                for synapse in self.synapses_out
            ])

        self.delta = self.__error * self.transfer_derivative(self.output)

    def update_weights(self):
        for synapse in self.synapses_in:

            synapse.weight += (self.__learning_rate
                               * self.delta * synapse.neuron_in.output)

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

            self.output = self.transfer(self.__activation)

        # input was directly provided to neuron, meaning this is a
        # first layer neuron. In this case, each Neuron has only one
        # input and one weight. This will be used to calculate its transfer
        # function
        else:
            if isinstance(network_input, list) and len(inputs) != 1:
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


class Network:

    def __init__(self, shape, learning_rate=0.1):
        # create layers of the neural net
        self.__learning_rate = learning_rate
        self.__layers = [[Neuron(learning_rate) for _ in range(
            layer_length)] for layer_length in shape]

        # fully connects each layer to the next
        for layer_index, layer in enumerate(self.__layers):

            # output layer has already been connected to by the previous
            # layer, and has no other layer to connect to
            if layer_index == len(self.__layers) - 1:
                break

            for neuron in layer:
                # connect neuron to each neuron of the next layer
                for target_neuron in self.__layers[layer_index + 1]:

                    synapse = Synapse(
                        neuron_in=neuron,
                        neuron_out=target_neuron
                    )

                    neuron.synapses_out.append(synapse)
                    target_neuron.synapses_in.append(synapse)

    def print_activations(self):
        for index, layer in enumerate(self.__layers):
            activations = ", ".join(
                ["%.4f" % neuron.activation for neuron in layer])
            print("Layer %i: [ %s ]" % (index, activations))

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
            raise ValueError(
                "Input not the same shape as input layer"
            )

        for layer_index, layer in enumerate(self.__layers):
            for neuron_index, neuron in enumerate(layer):

                # activate this neuron with only a single
                # input as opposed as the previous layer's outputs
                if layer_index == 0:
                    neuron.activate(network_input[neuron_index])
                else:
                    neuron.activate()

    def backwards_pass(self, expected_output):
        if len(expected_output) != len(self.__layers[-1]):
            raise ValueError(
                "Expected output not the same shape as actual output")

        for layer_index, layer in reversed(list(enumerate(self.__layers))):
            for neuron_index, neuron in enumerate(layer):
                # calculate output layer error based on the acutal expected values
                # as oppsed to the weight/error for the next connected layer
                if layer_index == len(self.__layers) - 1:
                    neuron.calculate_error(expected_output[neuron_index])
                    # print ("Expected: %.4f, actual: %.4f, delta: %.4f" %(expected_output[neuron_index],neuron.output, neuron.delta))
                else:
                    neuron.calculate_error()
                neuron.update_weights()


# derivative of the sigmoid function, d/dx(s(x)) = s(x)(1-s(x))
# derivative of the Cost wrt to the weight of layer l:
#   dC/dwl  = dzl/dwl * dal/dzl      * dC/dal
#   dC/dwl  = a(l-1)  * s(x)(1-s(x)) * 2(al - y)

# derivative of the Cost wrt to the bias of layer l:
#   dC/dwl = dzl/dbl * dal/dzl      * dC/dal
#   dC/dwl = 1       * s(x)(1-s(x)) * 2(al-y)

# where y is the desired value
# where zl = wl*al+bl
# where al = sigmoid(activation) = s(al)
# where C = = (al-y)^2
