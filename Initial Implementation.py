import random
import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def vectorise(label):
    label_vector = np.zeros(10)
    label_vector[label] = 1
    return label_vector

training_examples = [training_example.flatten() / 255 for training_example in x_train]
training_labels = [vectorise(training_label) for training_label in y_train]
test_examples = [test_example.flatten() / 255 for test_example in x_test]

training_set = list(zip(training_examples, training_labels))
test_set = list(zip(test_examples, y_test))

#################### SECTION BREAK ####################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

#################### SECTION BREAK ####################

class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.bias_vectors = [np.random.randn(layer_size) for layer_size in layer_sizes[1:]]
        self.weight_matrices = [np.random.randn(previous_layer_size, current_layer_size) for 
            (previous_layer_size, current_layer_size) in zip(layer_sizes[1:], layer_sizes[:-1])]

    def cost_derivative(self, output_activations, training_label):
        return (output_activations - training_label)
        
    def evaluate(self, test_set):
        return sum([np.argmax(self.feedforward(test_example)) == test_label for 
            (test_example, test_label) in test_set])
    
    def feedforward(self, activation_vector):
        for (bias_vector, weight_matrix) in zip(self.bias_vectors, self.weight_matrices):
            activation_vector = sigmoid(weight_matrix @ activation_vector + bias_vector)
        return activation_vector

    def stochastic_gradient_descent(self, training_set, test_set, epochs, 
        mini_batch_size, learning_rate): 
        
        for epoch in range(1, epochs + 1):
            random.shuffle(training_set)
            mini_batches = [training_set[index : index + mini_batch_size] for 
                index in range(0, len(training_set), mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.gradient_descent_step(mini_batch, learning_rate)

            print("Epoch {}: {} / {}".format(epoch, self.evaluate(test_set), len(test_set)))

    def gradient_descent_step(self, mini_batch, learning_rate):
        bias_grad_totals = [np.zeros(bias_vector.shape) for 
            bias_vector in self.bias_vectors]
        weight_grad_totals = [np.zeros(weight_matrix.shape) for 
            weight_matrix in self.weight_matrices]
        
        for (training_example, training_label) in mini_batch:
            (bias_grads, weight_grads) = self.backpropagation(training_example, training_label)
            
            bias_grad_totals = [bias_grad_total + bias_grad for 
                (bias_grad_total, bias_grad) in zip(bias_grad_totals, bias_grads)]
            weight_grad_totals = [weight_grad_total + weight_grad for 
                (weight_grad_total, weight_grad) in zip(weight_grad_totals, weight_grads)]

        self.bias_vectors = [bias_vector - learning_rate / len(mini_batch) * bias_grad_total for 
            (bias_vector, bias_grad_total) in zip(self.bias_vectors, bias_grad_totals)]
        self.weight_matrices = [weight_matrix - learning_rate / len(mini_batch) * weight_grad_total for 
            (weight_matrix, weight_grad_total) in zip(self.weight_matrices, weight_grad_totals)]

    def backpropagation(self, training_example, training_label):
        bias_grads = [np.zeros(bias_vector.shape) for 
            bias_vector in self.bias_vectors]
        weight_grads = [np.zeros(weight_matrix.shape) for 
            weight_matrix in self.weight_matrices]
        
        activation_vector = training_example
        activation_vectors = [activation_vector]
        weighted_sum_vectors = []
        
        for (bias_vector, weight_matrix) in zip(self.bias_vectors, self.weight_matrices):
            weighted_sum_vector = weight_matrix @ activation_vector + bias_vector
            weighted_sum_vectors.append(weighted_sum_vector)
            activation_vector = sigmoid(weighted_sum_vector)
            activation_vectors.append(activation_vector)
        
        delta_vector = self.cost_derivative(activation_vectors[-1], training_label) * \
            sigmoid_prime(weighted_sum_vectors[-1])
        
        bias_grads[-1] = delta_vector
        weight_grads[-1] = np.outer(delta_vector, activation_vectors[-2])
            
        for layer in range(-2, -self.num_layers, -1):
            delta_vector = self.weight_matrices[layer + 1].T @ delta_vector * \
                sigmoid_prime(weighted_sum_vectors[layer])
            bias_grads[layer] = delta_vector
            weight_grads[layer] = np.outer(delta_vector, activation_vectors[layer - 1])
            
        return (bias_grads, weight_grads)

#################### SECTION BREAK ####################

example_network = Network([784, 16, 16, 10])
example_network.stochastic_gradient_descent(training_set, test_set, epochs = 30, 
    mini_batch_size = 10, learning_rate = 3)