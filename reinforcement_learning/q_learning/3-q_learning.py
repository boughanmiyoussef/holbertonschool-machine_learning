#!/usr/bin/env python3
"""
Q-learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
    env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
    epsilon=1, min_epsilon=0.1, epsilon_decay=0.05
):
    """
    Train the agent using Q-learning algorithm
    Args:
        env is the FrozenLakeEnv instance
        Q is a numpy.ndarray containing the Q-table
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        epsilon is the initial threshold for epsilon greedy
        min_epsilon is the minimum value that epsilon should decay to
        epsilon_decay is the decay rate for updating epsilon between episodes
    Returns: Q, total_rewards
        Q is the updated Q-table
        total_rewards is a list containing the rewards per episode
    """
    total_rewards = []

    # Initialize the Q-table if not already initialized
    for episode in range(episodes):
        state = env.reset()
        # Handle the case where the state is a tuple
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0

        # Epsilon decay
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, done, truncated, info = env.step(action)
            done = done or truncated

            if isinstance(new_state, tuple):
                new_state = new_state[0]

            # Handle the case where the reward is zero and done is True
            if reward == 0 and done:
                reward = -1

            # Update the Q-value using the Q-learning formula
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            state = new_state
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)

    # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q, total_rewards
