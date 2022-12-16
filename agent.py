import os
import numpy as np
import pandas as pd
import utils

# import necessary libraries here - see example agent for more details

# Pickled models need to be externally loaded. Y
# Seeing as the .tar.gz is unpacked in a different location, and
# relative imports are difficult in Python.
# See example here using stable baselines: 
# https://gitlab.com/quantalytics-ai/chisl-hackathon-2022/-/blob/main/agent_example_1/agent.py#L13
AGENT_LOCATION = os.getenv('AGENT_LOCATION','./')

# Template agent
class Agent:

    def reward_function(
        self,
        navs: list,
    ):
        return utils.gordon(navs)

    def decision_function(
        self,
        observation: dict, 
        prev_reward: int, 
        info: dict,
    ):
        # Add RL agent's actions here (50 stocks)
        action = np.random.uniform(low=-0.02, high=0.02, size=(50,))
        # Always return action as an np.array of length observation space.
        return action
