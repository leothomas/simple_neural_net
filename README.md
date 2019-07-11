# simple_neural_net
An OOP neural net, created as an exercise in understanding gradient descent and back propagation

Many sources I've found online either gloss over gradient descent or describe it mathematically in terms of complex matrix operations that I find difficult to intutively understand, so I've made this OOP neural net (very inefficient) in an effort to gain a more intuitive understanding of what happens at the level of each neuron in terms of activation and backpropagation. 

I've trained this network against the sine function, and it seems the ends of the sine function are not "clamping down" correctly, and I'm having trouble understanding why this is the case. 

Images of tests performed on ~125 randomly selected inputs after every 100 epochs, are included for each network. Each folder name is prepended with the shape of the network that was used. 

All networks were trained with 500 epochs of 500 forward/backwards passes and a learning rate of 0.2

The [1, 128, 1] shape network was trained to fit to wavelenghts of the sin curve. It seems that it only move such as to reduce the error towards the left and right ends of the function, but couldn't "break" away from a very strongly linear shape
