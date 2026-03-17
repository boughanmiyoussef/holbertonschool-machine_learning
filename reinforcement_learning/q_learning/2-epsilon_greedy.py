#!/usr/bin/env python3
"""
Epsilon Greedy
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon Greedy
    Args:
        Q: numpy.ndarray of shape (s, a) containing the Q table
        state: integer representing the current state
        epsilon: float representing the epsilon to use for the calculation
    Return:
        the next action index
    """
    P = np.random.uniform(0, 1)
    if P < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])
