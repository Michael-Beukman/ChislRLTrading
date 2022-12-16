# %%
import sys
import gzip
import tarfile
import shutil
import pandas as pd
import numpy as np
import os
import diffusion as dif
import dill as pickle
from datetime import datetime, timedelta

from hacking_env import MyHackathonEnv as HackathonEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, SAC

# %% [markdown]
# ## Using stable baselines3
# Stable baselines does not allow its agents to be pickled, therefore an alternative method of packaging is used. In the agent.py file, the model is read upon initialisation, and used as an external model.
#
# You can train the stable baselines model, save it, and then load it by importing the agent file, and use the agent as normal.
#
# When packaging, ensure that you add the model and correctly specify the name.

# %% [markdown]
# ### 1 - Create agent

# %%
import numpy as np
import pandas as pd
import utils

from stable_baselines3.common.monitor import Monitor
import gym
data = pd.read_pickle("../train.pkl")

gym.register(
             id='MyEnv-v0',
            entry_point='hacking_env:MyHackathonEnv',
            kwargs= {'df': data}
    )

env = make_vec_env('MyEnv-v0', n_envs=1)
model = SAC('MlpPolicy', env, verbose=2)
model.set_env(env)

for i in range(0, 2):
    model.learn(total_timesteps=250_000, log_interval=1)
    model.save(f"saved_{i}")


for i in range(2, 12):
    model.learn(total_timesteps=10_000, log_interval=1)
    model.save(f"saved_{i}")
