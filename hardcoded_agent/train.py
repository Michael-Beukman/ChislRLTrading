# %%
import sys
import gzip
import tarfile
import shutil
import pandas as pd
import numpy as np
import diffusion as dif
import dill as pickle
from datetime import datetime, timedelta
from agent import Agent
import os
sys.path.append('../')
from hacking_env import HackathonEnv

# %% [markdown]
# ### 1 - Create agent

# %%
agent = Agent()

# %% [markdown]
# ### 2 - Train agent (if applicable) 

# %%
# agent.train()

# %% [markdown]
# ### 3 - Test agent

# %%

df = pd.read_pickle("../train.pkl")
env = HackathonEnv(df)

# 5 - Run simulation
t = 0
observation = env.reset()
# initial reward
reward = 0
info = {'sharpe': 1}
while True:
    break
    action = agent.decision_function(observation, reward, info)
    observation, reward, done, info = env.step(action)
    env.render()
    t+=1
    if done:
        print(f"Episode finsihed after {t+1} timesteps")
        break

status = "success"
print(info)
details = info['sharpe']

# %% [markdown]
# ### 4 - Package agent

# %%
with open(f'./agent.pkl', 'wb') as f:
    pickle.dump(agent, f)

# %%
files = ['./agent.py','./utils.py','agent.pkl']
with tarfile.open('./000010.tar.gz', 'w:gz') as tar:
    for file_source in files:
        tar.add(file_source, arcname=os.path.basename(file_source))

# %% [markdown]
# ### 5 - Submit!

# %% [markdown]
# 


