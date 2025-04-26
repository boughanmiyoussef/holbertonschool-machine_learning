#!/usr/bin/env python3
"""
Defines the Neuron class for binary classification tasks.
"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification with
    private attributes."""

    def __init__(self, nx):
        """
        Constructor for Neuron class.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for private attribute W."""
        return self.__W

    @property
    def b(self):
        """Getter for private attribute b."""
        return self.__b

    @property
    def A(self):
        """Getter for private attribute A."""
        return self.__A
