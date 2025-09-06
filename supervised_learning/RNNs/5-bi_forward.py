#!/usr/bin/env python3
"""
Bidirectional Cell Forward
"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden states
            o is the dimensionality of the outputs
        Attributes:
            Whf and bhf are for the hidden states in the forward direction
            Whb and bhb are for the hidden states in the backward direction
            Wy and by are for the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction
        Args:
            x_t is a numpy.ndarray of shape (m, i)
                m is the batch size for the data
            h_prev is a numpy.ndarray of shape (m, h)
        Returns:
            h_next: the next hidden state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(
            np.matmul(concat, self.Whf) + self.bhf
        )

        return h_next
