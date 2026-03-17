#!/usr/bin/env python3
"""
Initialize Q-table
"""
import numpy as np


def q_init(env):
    """
    Initialize the Q-table for the given environment.
    Args:
        env (gym.Env): The environment to initialize the Q-table for.
    Returns:
        np.ndarray: The initialized Q-table with shape (n_states, n_actions).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    return Q
