#!/usr/bin/env python3
"""
Gated Recurrent Unit (GRU)
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
        Args:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs
        The public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
            Wz and bz are for the update gate
            Wr and br are for the reset gate
            Wh and bh are for the intermediate hidden state
            Wy and by are for the output
        """
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step
        Argrs:
            h_prev is the previous hidden state
            x_t is the input data for the cell
        Returns:
            h_next is the next hidden state
            y_t is the output of the cell
        """
        # Concatenate previous hidden state and current input
        concat_h_x = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z_t = self.sigmoid(np.dot(concat_h_x, self.Wz) + self.bz)

        # Reset gate
        r_t = self.sigmoid(np.dot(concat_h_x, self.Wr) + self.br)

        # Intermediate hidden state
        concat_r_h_x = np.concatenate((h_prev * r_t, x_t), axis=1)

        # h_hat_t
        h_hat_t = np.tanh(np.dot(concat_r_h_x, self.Wh) + self.bh)

        # Final hidden state
        h_next = (1 - z_t) * h_prev + z_t * h_hat_t

        # Output
        y_t = np.dot(h_next, self.Wy) + self.by

        # Softmax activation
        y = self.softmax(y_t)

        # Return the next hidden state and the output
        return h_next, y

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function
        Args:
            x is the input data
        Returns:
            The sigmoid of x
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Softmax activation function
        Args:
            x is the input data
        Returns:
            The softmax of x
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
