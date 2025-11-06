
#!/usr/bin/env python3
import gymnasium as gym
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers import Adam
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

env = gym.make("Breakout-v4", render_mode="rgb_array", frameskip=1)
env = AtariPreprocessing(env)
env = GymnasiumObsWrapper(env)
nb_actions = env.action_space.n

obs = env.reset()
if isinstance(obs, (tuple, list)):
    obs = obs[0]

frames = np.array(obs)

print(len(env.step(env.action_space.sample())))

print(obs.shape)

nb_actions = env.action_space.n
print(f"Number of actions: {nb_actions}")

model = Sequential()
model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(nb_actions, activation='linear'))
print(obs.shape)

memory = SequentialMemory(limit=1000000, window_length=4)
print("Memory created with limit:", memory.limit)
print("Memory window length:", memory.window_length)

policy = EpsGreedyQPolicy()
print("type of policy:", type(policy))
print("Epsilon value:", policy.eps)

dqn = DQNAgent(
    model=model,
    memory=memory,
    policy=policy,
    nb_actions=nb_actions,
    nb_steps_warmup=50000,
    target_model_update=10000,
    gamma=0.99,
    train_interval=4,
    delta_clip=1.0
)

print("DQN Agent created with the following parameters:")
print(f"Model: {model}")
print(f"Memory: {memory}")
print(f"Policy: {policy}")
print(f"Number of actions: {nb_actions}")
print(f"Warmup steps: {dqn.nb_steps_warmup}")
print(f"Target model update: {dqn.target_model_update}")
print(f"Gamma: {dqn.gamma}")
print(f"Train interval: {dqn.train_interval}")
print(f"Delta clip: {dqn.delta_clip}")
print("Compiling the DQN Agent...")

dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
print("DQN Agent compiled successfully.")

dqn.fit(
    env,
    nb_steps=500000,
    visualize=False,
    verbose=2,
)
dqn.save_weights('dqn_weights.h5f', overwrite=True)

model.save('policy.h5')
