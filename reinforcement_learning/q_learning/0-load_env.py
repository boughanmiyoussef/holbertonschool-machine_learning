#!/usr/bin/env python3
"""
Module that loads the pre-made FrozenLake environment from gymnasium
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment from gymnasium.

    Args:
        desc: list of lists containing a custom description of the map,
              or None to use a pre-made or random map.
        map_name: string of a pre-made map name (e.g., '4x4', '8x8'),
                  or None for a random map.
        is_slippery: boolean determining if the ice is slippery.

    Returns:
        The loaded FrozenLake environment.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env
