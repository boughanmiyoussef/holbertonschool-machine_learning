#!/usr/bin/env python3
"""
This module defines the DeepNeuralNetwork class for binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Define a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Constructor for DeepNeuralNetwork class.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)  # Number of layers
        self.__cache = {}  # Dictionary to store forward propagation values
        self.__weights = {}  # Dictionary to store weights and biases

        # Initialize weights and biases
        for i in range(1, self.__L + 1):
            layer_size = layers[i - 1]
            prev_layer_size = nx if i == 1 else layers[i - 2]

            # He initialization weights
            self.__weights[f'W{i}'] = \
                np.random.randn(layer_size, prev_layer_size) \
                * np.sqrt(2 / prev_layer_size)
            self.__weights[f'b{i}'] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the number of cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the number of weights."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the deep neural network.
        """
        self.__cache['A0'] = X
        A = X
        for i in range(1, self.__L + 1):
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            Z = np.dot(W, A) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f'A{i}'] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y)
                                 * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = (A >= 0.5).astype(int)
        return predictions, cost
