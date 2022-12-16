import numpy as np
import pandas as pd
import utils
from stable_baselines3 import PPO, SAC
import os

# stable_baselines3 does not allow its models
# to be pickled. Therefore, they need to be
# externally loaded. Seeing as the .tar.gz
# is unpacked in a different location, and
# relative imports are difficult in Python.
AGENT_LOCATION = os.getenv('AGENT_LOCATION','./')
model = SAC.load(f"{AGENT_LOCATION}saved_model")
class Agent:
    def reward_function(
        self,
        navs: list,
    ):
        return utils.gordon(navs)

    def decision_function(
        self,
        observation: np.ndarray, 
        prev_reward: float, 
        info: dict,
    ):
        # Add complex logic here to determine what the agent has to do.
        action,_ = model.predict(observation.astype(np.float32))
        # Always return action as an np.array of length observation space.
        return action