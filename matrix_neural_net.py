import numpy as np


class Network:
    ACTIVATIONS = {
        'tanh': (
            lambda X: np.tanh(X),
            lambda X: 1. - np.square(X)
        ),
        'sin': (
            lambda X: np.sin(X),
            lambda X: -np.cos(X)
        ),
        'relu': (
            lambda X: np.maximum(X, 0, X),
            lambda X: np.greater(X, 0).astype(int)
        ),
        'sigmoid': (
            lambda X: 1. / (1. + np.exp(-X)),
            lambda X: X * (1. - X)
        ),
        'linear': (
            lambda X: np.array(
                [-1 if x < -1 else 1 if x > 1 else x for x in X]
            ),
            lambda X: np.array([0 if x < -1 or x > 1 else 1 for x in X])
        )
    }

    def __init__(self, shape, activation='tanh', output_activation='sigmoid', learning_rate=0.02):
        self.__shape = shape
        self.__weights = []
        self.__activations = []
        self.__biases = []

        for i, layer_length in enumerate(self.__shape):

            self.__activations.append(np.zeros(layer_length))

            if i != 0:
                self.__biases.append(
                    np.random.randn(layer_length) * 0.5
                )
                self.__weights.append(
                    np.random.randn(self.__shape[i-1], layer_length) * 0.5
                )

        self.__transfer, self.__transfer_derivative = self.ACTIVATIONS[activation]

        self.__output_transfer, self.__output_transfer_derivative = self.ACTIVATIONS[
            output_activation]

        self.__learning_rate = learning_rate

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
            raise ValueError("Input shape incorrect. Got len %s expected len %s" % (
                len(X), self.__shape[0]
            ))

        self.__activations[0] = X

        for i in range(0, len(self.__activations)-1):

            z = np.dot(self.__activations[i],
                       self.__weights[i]) + self.__biases[i]

            if i == len(self.__activations)-2:
                
                #print ("Z: %.4f, transfer: %.4f"%(z, self.__output_transfer(z)))
                
                self.__activations[i+1] = self.__output_transfer(z)

            else:
                self.__activations[i+1] = self.__transfer(z)
        
        #print ("Output: ", self.__activations[-1])

        return self.__activations[-1]

    def backward_pass(self, network_input, network_output, expected_output):
        if not isinstance(expected_output, np.ndarray):
            expected_output = np.array(expected_output)

        if not isinstance(network_input, np.ndarray):
            network_input = np.array(network_input)

        if len(expected_output) != self.__shape[-1]:
            raise ValueError("Expected output shape incorrect. Got len %s expected len %s" % (
                len(expected_output), len(self.__shape[-1])
            ))
        if len(network_input) != self.__shape[0]:
            raise ValueError("Input shape incorrect. Got len %s expected len %s" % (
                len(network_input), len(self.__shape[0])
            ))

        # Output error
        error = expected_output - network_output
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

            # error/ delta for the next layer of weights and biases
            error = np.dot(delta, self.__weights[i].T)
            delta = error * self.__transfer_derivative(self.__activations[i])

            # TODO: figure out how to avoid this reshape operation
            weight_update = weight_update.reshape(self.__weights[i].shape)

            self.__biases[i] += bias_update
            self.__weights[i] += weight_update

    def train_test(
        self, X, y, mode = "classify", 
        test_split = 0.2, learning_rate=None, 
        train_indices=None, test_indices=None, progress=None
    ):
        """
        progress is a callback function that will be invoked with: 
            progress(b:int, tsize:int)
        """

        # TODO: implement some sort of accuracy function for regression 

        if not train_indices and not test_indices:
            
            idx = np.arange(len(X))    
            np.random.shuffle(idx)
            test_indices = idx[:int(test_split*len(X))]
            train_indices = idx[int(test_split*len(X)):]
        
        # training phase
        correct = 0
        
        for i in range(len(train_indices)):

            output = self.forward_pass(X[train_indices[i]])
            expected = y[train_indices[i]]

            if mode == "classify":
                correct += int(np.argmax(output) == np.argmax(expected))

            if learning_rate == "step":
                step_size = int(len(train_indices)/20)
                if not i % step_size and i > 0: 
                    self.learning_rate = self.learning_rate/2
            
            self.backward_pass(
                network_input = X[train_indices[i]],
                network_output = output,
                expected_output = expected
            )
            if progress: 
                progress(b=i, tsize=len(train_indices))
        
        training_accuracy = correct/len(train_indices)
        
        # testing phase
        correct = 0  
        for i in range(len(test_indices)):
        
            output = self.forward_pass(X[train_indices[i]])
            expected = y[train_indices[i]]

            if mode == "classify":
                correct += int(np.argmax(output) == np.argmax(expected))
            
            if progress: 
                progress(b=i, tsize=len(test_indices    ))
            
            #print ("NN output: ", output)
            
            #print ("NN output: %i; Expected: %i" %(np.argmax(output), np.argmax(expected)))

        testing_accuracy = correct/len(test_indices)
    
        return training_accuracy, testing_accuracy
    
    def train_test_minibatch(
        self, X, y, mode="classify", 
        learning_rate=None, test_split=0.2, 
        batch_size=32, epochs=100, progress=None
    ):
        """
        progress is a callback function that will be invoked with: 
            progress(current:int, total:int)
        """

        idx = np.arange(len(X))    
        np.random.shuffle(idx)
        test_indices = idx[:int(test_split*len(X))]
        train_indices = idx[int(test_split*len(X)):]

        # training phase

        num_batches = int(len(train_indices)/batch_size)
        
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(epochs):
            
            back_prop_inputs = []
            correct = 0
            for i in range(len(train_indices)):
                network_input = X[train_indices[i]]
                output = self.forward_pass(network_input)
                expected = y[train_indices[i]]
                
                if mode == "classify":
                    correct += int(np.argmax(output) == np.argmax(expected))
                
                back_prop_inputs.append({
                    'network_input': network_input,    
                    'network_output': output,
                    'expected_output': expected
                })
                
                if not i % batch_size: 
                    for bp_input in back_prop_inputs:
                        self.backward_pass(**bp_input)
                    back_prop_inputs = []
            
            train_accuracies.append(correct/len(train_indices))
            if learning_rate == "step":
                step_size = int(epochs/20)
                if not i % step_size and i > 0: 
                    self.learning_rate = self.learning_rate/2
            # testing phase        
            correct = 0  
            for i in range(len(test_indices)):
            
                output = self.forward_pass(X[train_indices[i]])
                expected = y[train_indices[i]]

                if mode == "classify":
                    correct += int(np.argmax(output) == np.argmax(expected))
            
            #print ("NN output: %.4f --> %i; Expected: %i" %(output, int(output>0.5), expected))

            test_accuracies.append(correct/len(test_indices))
            
            if progress:
                progress(b=epoch, tsize=epochs)
        
        return train_accuracies, test_accuracies

    # KFold cross validation for hyper paramter tuning
    def kfold(self, X,y, k):
        testing = [] 
        training = []

        fold_size = int(len(X)/k)

        for i in range(k):
                    
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            idx = list(idx)

            test_idx = idx[i*fold_size: (i+1)*fold_size]
            train_idx = idx[:i*fold_size]
            train_idx.extend(idx[(i+1)*fold_size:])
            
            train_acc, test_acc = self.train_test(
                X,y, 
                test_indices = test_idx, 
                train_indices = train_idx
            )
            testing.append(test_acc)
            training.append(train_acc)
        
        return np.mean(training), np.mean(testing)


if __name__ == "__main__":
    model_params = {
        'shape': [11, 8, 6, 1],
        'activation': 'tanh',
        'output_activation': 'sigmoid',
        'learning_rate': 0.001
    }
    nn = Network(**model_params)

