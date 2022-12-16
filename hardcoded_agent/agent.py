import numpy as np
import pandas as pd
import utils

class Agent:
    def __init__(self) -> None:
        self.count = 0
        self.test = 0
            
    
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
        
        if len(info.get('history', [])) == 0 or len(info.get('history', [])) == 1:
            self.test = 0
            self.count = 0
        
        
        # Take the first 5 stocks, buy and hold. Then go on to the next ones, repeat.
        action = np.zeros(50)
        L = 5
        i = self.test % len(action // L)
        action[i * L : (i+1)*L] = 0.2
        self.count += 1
        
        if self.count % 1000 == 0:
            action *= -1
            self.test += 1
        
        # Always return action as an np.array of length observation space.
        return action