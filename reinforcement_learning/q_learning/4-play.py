#!/usr/bin/env python3
"""
Module to let the trained agent play an episode on FrozenLake
using the trained Q-table and rendering each step.
"""

import numpy as np

ACTION_NAMES = ["Left", "Down", "Right", "Up"]

def play(env, Q, max_steps=100):
    """
    Plays an episode using the Q-table (always exploiting).

    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the trained Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: total reward accumulated
        rendered_outputs: list of rendered board states as strings
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []

    nrow, ncol = env.unwrapped.desc.shape

    for _ in range(max_steps):
        # Choose the best action from Q-table
        action = np.argmax(Q[state])

        # Render current board
        board_str = ""
        for r in range(nrow):
            row_str = ""
            for c in range(ncol):
                pos = r * ncol + c
                ch = env.unwrapped.desc[r, c].decode()
                if pos == state:
                    row_str += f"`{ch}`"
                else:
                    row_str += ch
            board_str += row_str
            if r != nrow - 1:
                board_str += "\n"

        # Append the action taken
        board_str += f"\n  ({ACTION_NAMES[action]})"
        rendered_outputs.append(board_str)

        # Take the action
        next_state, reward, done, truncated, _ = env.step(action)
        total_rewards += reward
        state = next_state

        if done or truncated:
            break

    # Render final state (without appending an action)
    final_board = ""
    for r in range(nrow):
        row_str = ""
        for c in range(ncol):
            pos = r * ncol + c
            ch = env.unwrapped.desc[r, c].decode()
            if pos == state:
                row_str += f"`{ch}`"
            else:
                row_str += ch
        final_board += row_str
        if r != nrow - 1:
            final_board += "\n"
    rendered_outputs.append(final_board)

    return total_rewards, rendered_outputs
