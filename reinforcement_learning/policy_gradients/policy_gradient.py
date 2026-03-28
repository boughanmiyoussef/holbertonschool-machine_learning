#!/usr/bin/env python3
"""
Simple Policy function
"""
import numpy as np


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def policy(state, weight):
    """
    Compute the policy for a given state and weight.
    Args:
        state: The input state.
        weight: The weights of the policy.
    Returns:
        The action probabilities.
    """
    z = np.matmul(state, weight)
    return softmax(z)


def policy_gradient(state, weight):
    """
    Compute the Monte-Carlo policy gradient based on state and a weight matrix
    Args:
        state: matrix representing the current observation of the environment
        weight:  matrix of random weight
    Returns:
        The action and the gradient (in this order)
    """
    probabilities = policy(state, weight)

    # Reshape probabilities if they are in a 2D array [[p1, p2...]]
    probs = probabilities.flatten()

    action = np.random.choice(len(probs), p=probs)

    # Compute the gradient of the log-probability
    # For softmax, d(log_pi)/dw = state * (y_target - probabilities)
    d_softmax = np.zeros_like(probs)
    d_softmax[action] = 1
    d_log_pi = d_softmax - probs

    grad = np.outer(state, d_log_pi)

    return action, grad
