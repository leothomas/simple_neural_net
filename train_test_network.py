import numpy as np
import random
import time
from progress.bar import Bar
from simple_neural_net import Network
import matplotlib.pyplot as plt


def target_function(x):
    # add 1.0 to make all outputs positive
    # multiply input by 2*np.pi to force all outputs to
    # a [0,1] range
    # multiply by 0.5 to reduce all output from [0,2]
    # to [0,1]
    # return (1.0 + np.sin(2*2 * np.pi * x)) * 0.5

    return np.sin(2 * 2 * np.pi * x)
    # return np.cos(2 * np.pi * x)
    # return - (x)**2 + 1


def train_test_sin(network, num_passes=500, ratio=0.1, target_function=target_function):
    bar = Bar('Training', max=num_passes)

    # training step
    for _ in range(num_passes):
        # generate a random value within the input range
        x = np.random.uniform(0, 1.0)

        # pass through the network (lne function only takes one input)
        network.forward_pass([x])

        # calculate the expected output value
        expected_output = target_function(x)

        # backpropgate the expected value through the network
        network.backwards_pass([expected_output])

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

        expected_output = target_function(x)

        # added squared difference between the expected and the actual
        # to calculate loss function
        error += np.sqrt((network.output_layer[0].output - expected_output)**2)

    print("Error:%.4f " % error)
    # return inputs and output from testing phase for graphing
    return error, inputs, actual_outputs


if __name__ == "__main__":
    print("instantiating network")
    # TODO: Make learning rate adaptable (ie: as error goes down
    # epoch to epoch, reduce learning rate, to avoid bouncing
    # over global minima)

    # TODO: allow each neuron to have its own, customized activation
    # function

    network = Network(shape=[1, 10, 25, 1], learning_rate=0.01)

    errors = []
    num_epochs = 100

    # prepare a graph that will update with the result of the
    # testing phase after each epoch
    plt.ion()

    x = np.arange(0.0, 1.0, step=0.01)
    expected = target_function(x)

    fig, ax = plt.subplots()

    reference_plot = ax.plot(x, expected, label="Expected Values")
    test_plot = ax.scatter(x, expected, label="Network Output", c="r")

    ax.legend()

    for i in range(num_epochs):
        print("iteration %i out of %i" % ((i + 1), num_epochs))
        error, test_inputs, test_outputs = train_test_sin(
            network,
            num_passes=500, ratio=0.25,
            target_function=target_function
        )

        errors.append(error)
        actual = []

        # update data with newest test run
        test_plot.set_offsets([list(i)
                               for i in list(zip(test_inputs, test_outputs))])
        fig.canvas.draw()
        fig.canvas.flush_events()

        if i > 0 and i % 25 == 0:
            fig.savefig("./images/epoch_%ix500.png" % i)

    # save last epoch
    fig.savefig("./images/epoch_%ix500.png" % i)

    # stop interactive mode
    plt.ioff()

    # error from each epoch
    print(errors)

    # generate graph of loss function
    fig1, ax1 = plt.subplots()
    ax1.plot(errors)
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Sum Error")
    time.sleep(0.1)

    fig1.savefig("./images/loss_function_500x500.png")

    plt.show()
