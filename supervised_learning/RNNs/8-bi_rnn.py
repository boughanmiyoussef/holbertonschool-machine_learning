#!/usr/bin/env python3
"""
Bidirectional Recurrent Neural Network
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    forward propagation for a bidirectional RNN
    Args:
        bi_cell: instance of BidirectionalCell that will used for the forward
            propagation
        X: data to be used, shape (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state in the forward direction, shape (m, h)
            h: dimensionality of the hidden state
        h_t: initial hidden state in the backward direction, shape (m, h)
    Returns:
        H: all hidden states, shape (t, m, h * 2)
            h * 2: concatenated hidden states from both directions
        Y: output data, shape (t, m, o)
            o: dimensionality of the outputs
    """
    t, m, _ = X.shape
    _, h = h_0.shape

    # NOTE twice h: forward & backward
    H = np.zeros((t, m, h * 2))
    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))

    # Initial forward and backward hidden states
    h_f = h_0
    h_b = h_t

    for step in range(t):
        # Forward pass
        h_f = bi_cell.forward(h_f, X[step])

        # Store forward hidden state
        H[step, :, :h] = h_f

        # Backward pass (starting from the end)
        h_b = bi_cell.backward(h_b, X[-1 - step])

        # Store backward hidden state
        H[-1 - step, :, h:] = h_b

    Y = bi_cell.output(H)

    return H, Y
