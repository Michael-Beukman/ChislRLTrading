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

# Create the stable baselines model using the openAI environment
data = pd.read_pickle("../train.pkl")
env = HackathonEnv(data)
env = Monitor(env)
SAVE_NAME = 'saved_11'
shutil.copyfile(SAVE_NAME + ".zip", 'saved_model.zip')
model = SAC.load('saved_model')
# %%
# %%
from agent import Agent
agent = Agent()
# Run simulation
t = 0
observation = env.reset()
# initial reward
reward = 0
_ = None
while True:
    action = agent.decision_function(observation, _, _)
    observation, reward, done, info = env.step(action)
    env.render()
    t+=1
    if done:
        print(f"Episode finsihed after {t+1} timesteps")
        break

status = "success"
print(info)
details = info['sharpe']


with open(f'./agent.pkl', 'wb') as f:
    pickle.dump(agent, f)

files = ['./agent.py','./utils.py', 'agent.pkl', 'saved_model.zip']
with tarfile.open('./000040.tar.gz', 'w:gz') as tar:
    for file_source in files:
        tar.add(file_source, arcname=os.path.basename(file_source))
