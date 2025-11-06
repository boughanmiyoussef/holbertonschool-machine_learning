#!/usr/bin/env python3
import gymnasium as gym
from keras.models import load_model
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from gymnasium.wrappers import AtariPreprocessing
import numpy as np

class GymnasiumObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        if isinstance(observation, (tuple, list)):
            observation = observation[0]
        return np.array(observation)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, (tuple, list)):
            obs = obs[0]
        return obs

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            if isinstance(obs, (tuple, list)):
                obs = obs[0]
            return obs, reward, done, info
        elif len(result) == 5:
            obs, reward, terminated, truncated, info = result
            if isinstance(obs, (tuple, list)):
                obs = obs[0]
            done = terminated or truncated
            return obs, reward, done, info
        else:
            return result

env = gym.make("Breakout-v4", render_mode="human", frameskip=1)
env = AtariPreprocessing(env)
env = GymnasiumObsWrapper(env)

env.render = lambda mode=None: None

model = load_model("policy.h5", compile=False)

memory = SequentialMemory(limit=1000000, window_length=4)
policy = EpsGreedyQPolicy(eps=0.0)

dqn = DQNAgent(
    model=model,
    nb_actions=env.action_space.n,
    memory=memory,
    policy=policy,
    nb_steps_warmup=0,
    target_model_update=10000,
    gamma=0.99,
    train_interval=4,
    delta_clip=1.0
)
dqn.compile(optimizer="adam")

dqn.test(env, nb_episodes=5, visualize=True)
