#!/usr/bin/env python3
"""
Load the Environment
"""
import gymnasium as gym


def load_frozen_lake(
    desc=None, map_name=None, is_slippery=False, render_mode=None
):
    """
    Loads the pre-made FrozenLakeEnv evnironment from gymnasium
    Args:
        desc (list, optional): containing a custom description
            of the map to load for the environment
        map_name (str, optional): containing the pre-made map to load
        is_slippery (bool, optional): is a boolean to determine if
            the ice is slippery
    Returns:
        the environment
    """
    env = gym.make(
        'FrozenLake-v1',
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi"
    )
    return env
