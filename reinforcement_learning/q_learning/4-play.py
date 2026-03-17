#!/usr/bin/env python3
"""
Play
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Play the FrozenLake environment using the Q-table
    Args:
        env: the environment to play
        Q: the Q-table
        max_steps: maximum number of steps to take in the environment
    Returns:
        The total rewards for the episode and a list of rendered outputs
    """
    state, _ = env.reset()
    render_output = [env.render()]

    total_rewards = 0

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        render_output.append(env.render())
        total_rewards += reward
        if truncated or terminated:
            break

    return total_rewards, render_output
