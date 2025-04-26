#!/usr/bin/env python3
"""
This module defines the Neuron class for binary classification tasks.

"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification."""

    def __init__(self, nx):
        """
        Constructor for Neuron class.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')

        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
