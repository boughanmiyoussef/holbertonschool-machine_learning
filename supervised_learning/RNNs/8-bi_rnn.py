#!/usr/bin/env python3
"""
Performs forward propagation for a deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Args:
        rnn_cells: list of RNNCell instances of length l
        X: np.ndarray of shape (t, m, i) with input data
        h_0: np.ndarray of shape (l, m, h) with initial hidden states

    Returns:
        H: np.ndarray of shape (t + 1, l, m, h) containing hidden states
        Y: np.ndarray of shape (t, m, o) containing outputs
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]  # output dimension from the last RNNCell

    # Initialize hidden states and outputs
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    for step in range(t):
        x_t = X[step]

        for layer in range(l):
            rnn_cell = rnn_cells[layer]
            h_prev = H[step, layer]

            if layer == 0:
                x_in = x_t
            else:
                x_in = H[step + 1, layer - 1]

            h_next, y = rnn_cell.forward(h_prev, x_in)
            H[step + 1, layer] = h_next

        # Use the output from the last layer only
        Y[step] = y

    return H, Y
