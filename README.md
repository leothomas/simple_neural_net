# simple_neural_net
An OOP neural net, created as an exercise in understanding gradient descent and back propagation

Many sources I've found online either gloss over gradient descent or describe it mathematically in terms of complex matrix operations that I find difficult to intutively understand, so I've made this OOP neural net (very inefficient) in an effort to gain a more intuitive understanding of what happens at the level of each neuron in terms of activation and backpropagation. 

I've successfully trained this network against the sin function (Network shape: [1, 10, 25, 1], tanh activation function in hidden layers and ReLu for the output later). 

To train/test this model, run: `python3 train_test_network.py` (you'll need matplotlib and progressbar packages installed)

Here is the loss function graphed over 100 training epochs (of 500 forwards and backwards passes):
![Loss Function](/images/loss_function_500x500.png)

You can clearly see when the network abandoned a straight line a y=0 and started moving towards a sin shape, around iteration 55. 

Here is the result of the final training epoch: 
![Train Iteration](/images/epoch_99x500.png)

I'm sure some tweaks to activation function and a dynamically adapting learning rate would perfect the networks performance. 

I am currently attempting to using a genetic algorithm to teach this netwok to the play the game of snake. Each generation is comprised of 55 networks which each play a game of snake. The top 6 scoring networks are selected and crossed between themselves, by selecting pairs of networks, and using half of the biases and weights from one and half from another to create a new network. This is repeated (with some small, random mutations) until there is a new generation of 55 which then plays again. Currently the networks make no progress and their success is completely random. I think there may be and issue with the way the biases and weights are selected from one or the other to create new offsprings. 

The interesting thing about this method is that the execution is not actually that slow, since no errors are ever backpropagated. 

I would also like to revist the code for the network in a matrix approach to see how much more efficient it is to train a network in terms of matrix operations as opposed to memory objects. 
