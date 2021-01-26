#
#Tutorials and documentation :
#https://www.python-course.eu/neural_network_mnist.php
#https://www.tensorflow.org/api_docs
#https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow#step-1-%E2%80%94-configuring-the-project
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "data/"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",") 
test_data[:10]

test_data[test_data==255]
test_data.shape

n_input = 784 #First layer, input
n_hidden1 = 512
n_hidden2 = 256
n_hidden3 = 128
n_output = 10  # Last layer, output

learning_rate = 1e-4 # Parameter adjustment. larger learning rate -> faster convergence
n_iterations = 1000 # Number of iteration of the training phase
batch_size = 128 # Number of training example at each step
dropout = 0.5 # Chance that a unit has to be randomly eliminated at each step (prevent overfitting)

#TODO :
# Building graph
# Training and testing
# Documentation
# Report