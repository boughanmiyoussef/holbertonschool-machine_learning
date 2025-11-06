#!/usr/bin/env python3
"""
SARSA(λ) algorithm implementation.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determine action using epsilon-greedy policy.
    """
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state, :])
    else:
        return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(
    env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
    epsilon=1, min_epsilon=0.1, epsilon_decay=0.05
):
    """
    Implements the SARSA(λ) algorithm.
    Args:
        env: The environment to train on.
        Q: Initial action-value function.
        lambtha: The lambda parameter for eligibility traces.
        episodes: Number of episodes to train.
        max_steps: Maximum steps per episode.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Initial exploration rate.
        min_epsilon: Minimum exploration rate.
        epsilon_decay: Decay rate for epsilon.
    Returns:
        Q, the updated Q table
    """
    initial_epsilon = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        eligibility = np.zeros_like(Q)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            TD_error = (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            eligibility *= gamma * lambtha
            eligibility[state, action] += 1

            Q += alpha * TD_error * eligibility

            state, action = next_state, next_action

            if terminated or truncated:
                break

        epsilon = (
            min_epsilon + (initial_epsilon - min_epsilon)
            * np.exp(-epsilon_decay * ep)
        )

    return Q
