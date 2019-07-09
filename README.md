# simple_neural_net
An OOP neural net, created as an exercise in understanding gradient descent and back propagation

Many sources I've found online either gloss over gradient descent or describe it purely mathematically in terms of complex matrix operatins that I find difficule to intutively understand, so I've made this example in a purely OOP fashion (very inefficient) in an effor to gain a more intuitive understanding of what happens at the level of each neuron in terms of activation and backpropagation. 

I've trained this network against the sin function, and it seems the ends of the sin function are "clamping down" correctly, and I'm having trouble understanding why this is the case. 

The images included in this repository are an example of a testing pass through the network with ~125 test points, randomly distributed from 0 to 1, and a graph of the loss function. 

These were generated after training the model for 500 epochs, each epoch with 500 forwards/backwards passes, and a learning rate of 0.2.

The network has an input and an output layer of length 1 (sin(input) = output), and a fully connected layer with 32 nodes

I've tried different, fully connected networks such as 1, 128, 1 and 1, 16, 16, 1, whithout much more success

