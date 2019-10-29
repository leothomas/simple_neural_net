# simple_neural_net
I have created an OOP neural network, as an exercise in understanding gradient descent and back propagation

Many sources I've found online either gloss over gradient descent or describe it mathematically purely in terms of matrix operations that I find difficult to intutively understand, so I've made this OOP neural net (very inefficient) in an effort to gain a more intuitive understanding of what happens at the level of each neuron in terms of activation and backpropagation. 

I've successfully trained this network against the sin function (Network shape: [1, 10, 25, 1], tanh activation function in hidden layers and Linear activation for the output later). 

To train/test this model, run: `python3 train_test_network.py` (you'll need matplotlib and progressbar packages installed)

Here is the loss function graphed over 100 training epochs (of 500 forwards and backwards passes):
![Loss Function](/images/loss_function_500x500.png)

Clearly the network is performant and correctly learns to approximate the sine function. 

Here is the result of the final test run: 
![Train Iteration](/images/epoch_99x500.png)

I'm sure some tweaks to the activation functions and a dynamically adapting learning rate would perfect the network's performance. 

I am currently attempting to using a genetic algorithm to teach this netwok to the play the game of snake. The algorithm works by creating successive generations of 55 networks which each play a game of snake. The top 6 scoring networks are selected and crossed between themselves, by selecting pairs of networks and using half of the biases and weights from one and half from another to create a new network. This is repeated (with some small, random mutations) until there is a new generation of 55 networks which then plays again. 

To see this process, execute: `python3 snake_simple.py` (the game of snake will play in the comand line, using the curses package)

Currently the networks make no progress and their success is completely random. I think there may be an issue with the way the biases and weights are selected from one or the other to create new offsprings. 

To see the games of snake being played as the algorithm trains, change `visual=False` to `visual=True` in lines 430 and 439. 

The OOP networks are quite slow to train and memory intensive for such an application (some online sources say to try generations of 1000, 2000, or 2500 individuals). So I've made another class with the same method signatures as the OOP neural network, which will implement the forward/backward propagation in terms of matrix operations. This executes much more efficiently and can handle more complex network shapes. To train and test this network against a simple sine function, switch lines 5 and 6 and comment out lines 79-81, and run `python3 train_test_network.py`. 

To see this network's performance against the MNIST handwritten digit recognition dataset, take a look at this [jupyter notebook](/MNIST.ipynb). This network acheived an ~80% accuracy which can easily be improved upon with some of the optimizations listed at the bottom of the notebook. 

If you spot any errors potential improvements, please reach out!
