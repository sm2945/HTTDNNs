import random
import numpy as np
from keras.datasets import mnist

(training_exmpls, training_lbls), (test_exmpls, test_lbls) = mnist.load_data()

def vectorise(label):
  label_vector = np.zeros((10, 1))
  label_vector[label] = 1
  return label_vector

training_exmpls = [training_exmpl.reshape(784, 1) / 255 for
  training_exmpl in training_exmpls]
training_lbls = [vectorise(training_lbl) for
  training_lbl in training_lbls]
test_exmpls = [test_exmpl.reshape(784, 1) / 255 for
  test_exmpl in test_exmpls]

training_set = list(zip(training_exmpls, training_lbls))
test_set = list(zip(test_exmpls, test_lbls))

# training_set = random.sample(training_set, 600)
# test_set = random.sample(training_set, 100)

#################### SECTION BREAK ####################

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
  return sigmoid(x) * (1 - sigmoid(x))

def cost_derivative(output_activations, training_lbl):
  return (output_activations - training_lbl) / \
    (output_activations * (1 - output_activations))

#################### SECTION BREAK ####################

class Network:
  def __init__(self, lyr_sizes):
    self.lyr_sizes = lyr_sizes
    self.num_lyrs = len(lyr_sizes)
    self.bias_vctrs = [np.random.randn(lyr_size, 1) for lyr_size in lyr_sizes[1:]]
    self.weight_mtrcs = [np.random.normal(0, 1 / np.sqrt(prvs_lyr_size),
      (crrnt_lyr_size, prvs_lyr_size)) for
        (prvs_lyr_size, crrnt_lyr_size) in zip(lyr_sizes[:-1], lyr_sizes[1:])]

  def feedforward(self, activation_vctr):
    for (bias_vctr, weight_mtrx) in zip(self.bias_vctrs, self.weight_mtrcs):
      activation_vctr = sigmoid(weight_mtrx @ activation_vctr + bias_vctr)
    return activation_vctr

  def eval(self, test_set):
    return sum([np.argmax(self.feedforward(test_exmpl)) == test_lbl for
      (test_exmpl, test_lbl) in test_set])
    
  def cross_entropy_cost(self, training_set):
    running_total = 0
    for (training_exmpl, training_lbl) in training_set:
      output_activations = self.feedforward(training_exmpl)
      for index in range(len(output_activations)):
        y_i, a_i = training_lbl[index], output_activations[index]
        running_total += -y_i * np.log(a_i) - (1 - y_i) * np.log(1 - a_i)
    return (running_total / len(training_set))[0]

  def stochastic_gradient_descent(self, training_set, test_set, max_epochs,
    mini_btch_size, eta, lmbda, max_epcohs_wtht_imprvmnt): 

    epoch = 1
    epcohs_wtht_imprvmnt = 0
    peak_accuracy = 0
    
    while epoch <= max_epochs and epcohs_wtht_imprvmnt <= max_epcohs_wtht_imprvmnt:
      random.shuffle(training_set)
      mini_btches = [training_set[index : index + mini_btch_size] for
        index in range(0, len(training_set), mini_btch_size)]

      for mini_btch in mini_btches:
        self.gradient_descent_step(mini_btch, eta, lmbda)

      accuracy = self.eval(test_set)
      if accuracy <= peak_accuracy:
        epcohs_wtht_imprvmnt += 1
      else:
        epcohs_wtht_imprvmnt = 0
        peak_accuracy = accuracy
      
      print("Epoch {}: {} / {}".format(epoch, accuracy, len(test_set)))
      print("Cost: {}".format(self.cross_entropy_cost(training_set)))
      print("Epochs without improvement: {}\n".format(epcohs_wtht_imprvmnt))
      epoch += 1

  def gradient_descent_step(self, mini_btch, eta, lmbda):
    bias_grad_ttls = [np.zeros(bias_vctr.shape) for
      bias_vctr in self.bias_vctrs]
    weight_grad_ttls = [np.zeros(weight_mtrx.shape) for
      weight_mtrx in self.weight_mtrcs]

    for (training_exmpl, training_lbl) in mini_btch:
      (bias_grads, weight_grads) = self.backprop(training_exmpl, training_lbl)

      bias_grad_ttls = [bias_grad_ttl + bias_grad for
        (bias_grad_ttl, bias_grad) in zip(bias_grad_ttls, bias_grads)]
      weight_grad_ttls = [weight_grad_ttl + weight_grad for
        (weight_grad_ttl, weight_grad) in zip(weight_grad_ttls, weight_grads)]

    self.bias_vctrs = [bias_vctr - eta / len(mini_btch) * bias_grad_ttl for
      (bias_vctr, bias_grad_ttl) in zip(self.bias_vctrs, bias_grad_ttls)]
    self.weight_mtrcs = [weight_mtrx * (1 - eta * lmbda / len(training_set)) - \
      eta / len(mini_btch) * weight_grad_ttl for
        (weight_mtrx, weight_grad_ttl) in zip(self.weight_mtrcs, weight_grad_ttls)]

  def backprop(self, training_exmpl, training_lbl):
    bias_grads = []
    weight_grads = []

    activation_vctr = training_exmpl
    activation_vctrs = [activation_vctr]
    weighted_sum_vctrs = []

    for (bias_vctr, weight_mtrx) in zip(self.bias_vctrs, self.weight_mtrcs):
      weighted_sum_vctr = weight_mtrx @ activation_vctr + bias_vctr
      weighted_sum_vctrs.append(weighted_sum_vctr)
      activation_vctr = sigmoid(weighted_sum_vctr)
      activation_vctrs.append(activation_vctr)

    delta_vctr = cost_derivative(activation_vctrs[-1], training_lbl) * \
      sigmoid_prime(weighted_sum_vctrs[-1])

    bias_grads.insert(0, delta_vctr)
    weight_grads.insert(0, delta_vctr @ activation_vctrs[-2].T)
  
    for layer in range(-2, -self.num_lyrs, -1):
      delta_vctr = self.weight_mtrcs[layer + 1].T @ delta_vctr * \
        sigmoid_prime(weighted_sum_vctrs[layer])
      bias_grads.insert(0, delta_vctr)
      weight_grads.insert(0, delta_vctr @ activation_vctrs[layer - 1].T)
  
    return (bias_grads, weight_grads)

#################### SECTION BREAK ####################

example_network = Network([784, 30, 10])
example_network.stochastic_gradient_descent(training_set, test_set, max_epochs = 30, 
  mini_btch_size = 10, eta = 0.25, lmbda = 6, max_epcohs_wtht_imprvmnt = 10)