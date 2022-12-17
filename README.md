# Chisl RL Hackathon -- Winning Solution

This repository contains the winning solution for the [Chisl](https://www.chislgroup.com/) RL Trading Hackathon, held at the IndabaX, South Africa, 2022.


## Problem
The problem is to trade stocks on the market. This is formulated as a reinforcement learning problem, where the states are information about 50 stocks, and the action space is a 50-dimensional continuous vector (between -1 and 1), representing the percentage of your total net worth you want to buy/sell of each stock.


## Solution
Here are the solutions. I have included the checkpoint files themselves as well, so the training does not need to be repeated.
### RL
This is the best-performing solution, but it takes a while to train.

This solution largely used a standard RL agent, based on the sample code provided. The specific changes were:
- Use SAC instead of PPO, as it is generally high-performing in continuous control settings.
- Train for around 60 times longer than the default code had, roughly 600000 steps.
- Some environmental changes (as training on the default environment for 12+ hours showed no improvement). Most of these could be achieved by having a wrapper class around the current environment, or manually altering the stable baselines code, but the easiest was just to modify the code itself.
  - Alter the reward function to optimise the Sharpe ratio directly.
  - Make the action space between -0.1 and +0.1, to constrain the agent.
  - Modify the environment to stop episodes early, as the default length was very long, leading to slow training.
  - Replace the try/except checks with if statements ([as they can be quite expensive](https://stackoverflow.com/questions/2522005/cost-of-exception-handlers-in-python)).
#### Reproduction
To reproduce this, you can do:

```
cd rl_agent
python train.py
python submit.py
```

### RL - Seeded
Now, the above RL agent did not set the random seed, so results may vary. The code in `rl_agent_seeded` does and will thus give the same results when run multiple times. The actual training code, however, is identical to that in `rl_agent`.
To reproduce this, you can run:

```
cd rl_agent_seeded
python train.py
python submit.py
```

### Hand-Coded
This solution was relatively simple, and acts as a baseline. The approach here was to take the first 5 stocks, and buy 20% of each. Then, 1000 steps later, sell all of these stocks, go on to the next 5, spend 20% of your net worth on each, and repeat.

#### Reproduction
To reproduce this result, you can do the following:
```
cd hardcoded_agent
python train.py
```

## Environment Setup
You can run the following to get started

```bash
conda create -n rl python=3.9 pandas numpy matplotlib
conda activate rl
pip install stable_baselines3 torch diffusion dill gym
```

Or, after creating the environment, you can run `pip install -r requirements.txt`. Finally, if none of these work, you can just install the environment as instructed by the [original code](https://gitlab.com/quantalytics-ai/chisl-hackathon-2022.git).

## Notes
While the above RL-based solution performed the best, there was a mistake, and the wrong version of the code was selected as the final chosen solution. This was not intended, but I have provided code for both solutions.

Most of this code was taken from the sample submission found here: https://gitlab.com/quantalytics-ai/chisl-hackathon-2022.git, so it was not written by me.