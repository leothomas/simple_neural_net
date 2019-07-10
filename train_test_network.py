import numpy as np
import random
import time
from progress.bar import Bar
from simple_neural_net import Network
import matplotlib.pyplot as plt


def train_test_ln(network, num_passes=500, ratio=0.1):
    bar = Bar('Training', max=num_passes)

    # training step
    for _ in range(num_passes):
        # generate a random value within the input range
        x = np.random.uniform(0, 1.0)

        # pass through the network (lne function only takes one input)
        network.forward_pass([x])

        # calculate the expected output value
        expected = (1.0 + np.sin(2*np.pi * x)) * 0.5

        # backpropgate the expected value through the network
        network.backwards_pass([expected])

        bar.next()
    bar.finish()

    # testing step

    error = 0.0
    inputs = []
    actual_outputs = []

    for _ in range(int(num_passes * ratio)):
        # generate random value within input range
        x = np.random.uniform(0, 1.0)

        # pass through the network
        network.forward_pass([x])
        # store input value for testing purposes
        inputs.append(x)
        # store output value for graphing purposes
        actual_outputs.append(network.output_layer[0].output)

        # added squared difference between the expected and the actual
        # to calculate loss function
        error += np.sqrt((network.output_layer[0].output -
                          0.5 * (1.0 + np.sin(2*np.pi * x)))**2)
       

    print("Error:%.4f " % error)
    # return inputs and output from testing phase for graphing
    return error, inputs, actual_outputs


if __name__ == "__main__":
    print("instantiaing network")
    # TODO: Make learning rate adaptable (ie: as error goes down
    # epoch to epoch, reduce learning rate, to avoid bouncing
    # over global minima)

    # TODO: allow each neuron to have its own, customized activation
    # function

    network = Network(shape=[1, 32, 1], learning_rate=0.2)

    errors = []
    num_epochs = 500

    # prepare a graph that will update with the result of the
    # testing phase after each epoch
    plt.ion()
    x = np.arange(0.0, 1.0, step=0.01)
    expected = 0.5 * (1.0 + np.sin(2*np.pi * x))
    fig, ax = plt.subplots()
    reference_plot = ax.plot(x, expected, label="Expected Values")
    test_plot = ax.scatter(x, expected, label="Network Output", c="r")
    ax.legend()

    for i in range(num_epochs):
        print("iteration %i out of %i" % ((i + 1), num_epochs))
        error, test_inputs, test_outputs = train_test_ln(
            network, num_passes=500, ratio=0.25)

        errors.append(error)
        actual = []

        # update data with newest test run
        test_plot.set_offsets([list(i)
                               for i in list(zip(test_inputs, test_outputs))])
        fig.canvas.draw()
        fig.canvas.flush_events()

        if i > 0 and i % 100 == 0:
            fig.savefig("./epoch_%ix500.png" % i)

    # stop interactive mode
    plt.ioff()

    # generate a graph over the input space to visualize the output
    # of the network
    #outputs = []
    # for i in x:
    #    network.forward_pass([i])
    #    outputs.append(network.output_layer[0].output)

    #fig0, ax0 = plt.subplots()
    #reference_plot = ax0.plot(x, expected, label="Expected Values")
    #test_plot = ax0.plt(x, outputs, label="Network Output")

    # error from each epoch
    print(errors)

    # generate graph of loss function
    fig1, ax1 = plt.subplots()
    ax1.plot(errors)
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Sum Error")
    time.sleep(0.1)

    fig1.savefig("./loss_function_500x500.png")

    plt.show()
