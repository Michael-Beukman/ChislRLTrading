import os
import sys
import gzip
import shutil
import pandas as pd
from dill import loads, load
from hacking_env import HackathonEnv

def main(location: str, data: str):
    """
    Main function to test and verify agent class will run upon
    submission.

    Steps of main:
        1 - Unpack gzip file into agent/ directory
        2 - Verify agent.py template structure
        3 - Rehydrate agent object
        4 - Instantiate environment
        5 - Run simulation
    
    Args:
        location (string): location of gzip file to be unpacked
        data (string): location of pandas data pickle to be loaded
    Returns:
        score (float): overall score for env
    """
    if not os.path.isdir('./agent/'):
        os.makedirs('./agent')

    # 1 - Unpack gzip file into agent/ directory
    shutil.unpack_archive(filename=location, extract_dir='./agent')
    
    # 2 - Verify agent.py template structure
    #TODO
    os.environ['AGENT_LOCATION'] = "./agent/"

    # 3 - Rehydrate agent object
    sys.path.append('./agent/')
    map(__import__, ['agent','utils'])

    agent = None
    with open('./agent/agent.pkl','rb') as f:
        agent = load(f)

    df = pd.read_pickle(data)
    env = HackathonEnv(df)

    # Run simulation
    t = 0
    observation, info = env.reset()
    # initial reward
    reward = 0
    while True:
        action = agent.decision_function(observation, reward, info)
        observation, reward, done, info = env.step(action)
        env.render()
        t+=1
        if done:
            print(f"Episode finsihed after {t+1} timesteps")
            break

    final_score = info['sharpe']
    if pd.isna(details):
        details = 0
    print(f"Final score is:", final_score)
        

if __name__ == "__main__":
    main("./agent-archive.tar.gz", "./data.pkl")