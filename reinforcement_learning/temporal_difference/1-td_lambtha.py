#!/usr/bin/env python3
"""
TD(λ)
"""
import numpy as np


def td_lambtha(
    env, V, policy, lambtha, episodes=5000,
    max_steps=100, alpha=0.1, gamma=0.99
):
    """
    Performs the TD(λ) algorithm
    Args:
        env: the environment
        V: a numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in an integer state and returns the action
        lambtha: the eligibility trace factor
        episodes: the number of episodes to train over
        max_steps: the maximum number of steps per episode
        alpha: the learning rate
        gamma: the discount factor
    Returns:
        V: the updated value estimate
    """
    for ep in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros_like(V)
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, trucated, _ = env.step(action)

            # Calculate the TD error
            delta = reward + (gamma * V[next_state] - V[state])

            eligibility[state] += 1

            # Update the value function
            V += alpha * delta * eligibility

            # Update the eligibility trace
            eligibility *= gamma * lambtha

            # Update the state
            state = next_state

            if done or trucated:
                break
    return V
