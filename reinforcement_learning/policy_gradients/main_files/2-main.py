#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
train = __import__('train').train

def set_seed(env, seed=0):
    env.reset(seed=seed)
    np.random.seed(seed)
    random.seed(seed)

env = gym.make('CartPole-v1')
set_seed(env, 0)

scores = train(env, 10000)

plt.plot(np.arange(len(scores)), scores)
plt.show()
env.close()
