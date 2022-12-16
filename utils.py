import numpy as np

def gordon(navs, kappa=1e-1):
    """
    Reward function as suggested by Gordon:
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3015609
    Args:
        navs (list): array of net asset values over each setp
        kappa (float): hyperparam between 0 and 1; closer to 0 is risk neutral
    Returns:
        reward (float): last return
    """
    # returns (numpy.ndarray): percentage returns
    returns = np.diff(navs) / navs[:-1]  #  convert return to decimal percentage
    returns[np.bitwise_not(np.isfinite(returns))] = 0.
    reward = returns[-1] - 0.5 * kappa * (returns[-1] ** 2)
    return reward
