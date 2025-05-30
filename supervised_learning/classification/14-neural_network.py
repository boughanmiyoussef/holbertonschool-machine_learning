#!/usr/bin/env python3
"""
This module defines the NeuralNetwork class for binary classification
with one hidden layer.
"""
import numpy as np


class NeuralNetwork:
    """
    Define a neural network with one hidden layer performing binary
    classification.
    """

    def __init__(self, nx, nodes):
        """
        Constructor for NeuralNetwork class.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for private attribute W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter for private attribute b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter for private attribute A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter for private attribute W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter for private attribute b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter for private attribute A2."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network using a
        sigmoid activation function.
        """
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))  # Sigmoid hidden layer.
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))  # Sigmoid output layer.
        return self.__A1, self.__A2

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
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        predictions = (A2 >= 0.5).astype(int)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform one pass of gradient descen on the neural network.
        """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network using forward propagation and gradient
        descent.
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')

        costs = []
        for _ in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        return self.evaluate(X, Y)
