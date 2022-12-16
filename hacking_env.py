import gym
from gym import spaces
import numpy as np
from datetime import datetime
INITIAL_NOMINAL = 100_000
# if this hcanges then number of ptf observations need to change in _get_obs
WINDOW_SIZE = 5 

class InvalidTradeException(Exception):
    pass


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

class HackathonEnv(gym.Env):
    """Custom hackathon environment"""
    metadata = {'render.modes':['human']}

    def __init__(
        self, 
        df,
        max_leverage=5,
    ):
        super(HackathonEnv, self).__init__()

        # Meta env info
        self.window_size = WINDOW_SIZE
        self.done = False
        self.df = df
        self.num_stocks = self.df.shape[1]
        self.max_steps = self.df.shape[0] - 1
        self.max_leverage = max_leverage
        self.feedback = ""
        self.max_episode_length = df.shape[0]

        # init
        # add + 1 for the portoflio observations
        self.shape = (WINDOW_SIZE, self.num_stocks + 1)
        self.cash = INITIAL_NOMINAL
        self.equity = 0
        self.current_step = 0 + WINDOW_SIZE
        self.net_worths = [INITIAL_NOMINAL]
        self.history = []
        self.initial_balance = INITIAL_NOMINAL
        self.last_reward = 0
        self.num_trades = 0
        self.purchase_prices = np.zeros(self.num_stocks)

        self.positions = np.array([0]*self.num_stocks)
        self.shares = np.array([0]*self.num_stocks)

        # use a standard reward function
        self.reward_function = gordon
        # TODO: Incorporate feature func in obs space
        self.features_function = self.def_features

        # Create the action space, between -1 and 1
        # to be able to short and long.
        # value is % of nominal funds to pull out
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_stocks,),
            dtype=np.float64
        )

        # Create the observation space, 
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.shape,
            dtype=np.float64
        )

    def reset(self, initial_cash=INITIAL_NOMINAL):
        # should return (observation, info)
        # Reset the state of the environment to an initial state
        self.cash = initial_cash
        self.equity = 0
        self.positions = np.array([0]*self.num_stocks)
        self.shares = np.array([0]*self.num_stocks)
        self.done = False
        self.history = []
        self.current_step = 0 + WINDOW_SIZE
        self.last_reward = 0
        self.feedback = ""
        self.net_worths = [INITIAL_NOMINAL]
        self.initial_balance = INITIAL_NOMINAL
        self.purchase_prices = np.zeros(self.num_stocks)
        self.num_trades = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs#, info


    def step(self, action):
        # should return (observation, reward, done, info)
        self.feedback = ""
        def inner(action):
                # plus one as assume excute trade the next day
                self.purchase_prices = self.df.iloc[self.current_step+1]

                if sum(action) < -1 or 1 < sum(action):
                    # invalid action, cannot buy more than the nominal you have
                    self.feedback = "Tried to allocate more than 100% of nav"
                    self.last_reward = -100; return
                
                if not np.allclose(self.weights, action, 1e-5):
                    
                    shares_adjustment = np.fix((action * self.nav)/self.purchase_prices) - self.shares
                    action_cost = sum((shares_adjustment * self.purchase_prices))
                    
                    new_shares = self.shares + shares_adjustment
                    new_cash = self.cash - action_cost
                    new_equity = sum(new_shares * self.purchase_prices)
                    new_leverage = sum(np.abs(new_shares * self.purchase_prices))/(new_equity+new_cash)
                    
                    if new_cash < 0:
                        self.feedback = "Not enough cash to execute trade"
                        self.last_reward = -100; return
                    
                    if self.max_leverage < new_leverage:
                        self.feedback = "Too much leverage"
                        self.last_reward = -100; return

                    # Commit trade
                    self.shares = new_shares
                    self.cash = new_cash
                    self.equity = sum((self.shares * self.purchase_prices))
                    self.positions = action
                    self.num_trades += 1

                self.net_worths.append(self.nav)
                self.last_reward = self._evaluate_reward()
        inner(action)
        self.history.append(action)
        self.current_step += 1
        self.done = self._done()

        return self._get_obs(), self.last_reward, self.done, self._get_info()


    def stepslow(self, action):
        # This is the original step function
        # should return (observation, reward, done, info)
        self.feedback = ""

        try:
            # plus one as assume excute trade the next day
            self.purchase_prices = self.df.iloc[self.current_step+1]

            if sum(action) < -1 or 1 < sum(action):
                # invalid action, cannot buy more than the nominal you have
                self.feedback = "Tried to allocate more than 100% of nav"
                raise InvalidTradeException
            
            if not np.allclose(self.weights, action, 1e-5):
                
                shares_adjustment = np.fix((action * self.nav)/self.purchase_prices) - self.shares
                action_cost = sum((shares_adjustment * self.purchase_prices))
                
                new_shares = self.shares + shares_adjustment
                new_cash = self.cash - action_cost
                new_equity = sum(new_shares * self.purchase_prices)
                new_leverage = sum(np.abs(new_shares * self.purchase_prices))/(new_equity+new_cash)
                
                if new_cash < 0:
                    self.feedback = "Not enough cash to execute trade"
                    raise InvalidTradeException
                
                if self.max_leverage < new_leverage:
                    self.feedback = "Too much leverage"
                    raise InvalidTradeException

                # Commit trade
                self.shares = new_shares
                self.cash = new_cash
                self.equity = sum((self.shares * self.purchase_prices))
                self.positions = action
                self.num_trades += 1

            self.net_worths.append(self.nav)
            self.last_reward = self._evaluate_reward()
        except InvalidTradeException as ex:
            self.last_reward = -100
        except Exception as ex:
            self.feedback = f"{ex}"
        finally:
            self.history.append(action)
            self.current_step += 1
            self.done = self._done()

            return self._get_obs(), self.last_reward, self.done, self._get_info()

    @property
    def nav(self):
        return self.equity + self.cash
    
    @property
    def weights(self):
        return self.shares * self.purchase_prices / self.nav

    @property
    def leverage(self):
        current_prices = self.df.iloc[self.current_step]
        return (sum(np.abs(self.shares * current_prices))/self.nav)

    def sharpe(self, rf=0.05): # lower risk free ideally should be 7%
        # returns (numpy.ndarray): percentage returns
        returns = np.diff(self.net_worths) / self.net_worths[:-1]  #  convert return to decimal percentage
        returns[np.bitwise_not(np.isfinite(returns))] = 0.
        stdev = np.std(returns)
        rf_daily =  (1 + rf) ** (1/365) - 1
        sharpe = (np.mean(returns) - rf_daily) / stdev
        sharpe = sharpe * np.sqrt(252) # normalise by num trading days
        return sharpe

    def def_features(self, loookback=WINDOW_SIZE):
        return self.df.pct_change(1).values[self.current_step - WINDOW_SIZE:self.current_step]

    def _evaluate_reward(self):
        return self.reward_function(self.net_worths)

    def _evaluate_features(self):
        return self.features_function(self.df)

    def _get_obs(self):
        # return the observation of the current environment
        # sacled portfolioo observations (column vector)
        ptf_obs = np.vstack([
            self.nav / INITIAL_NOMINAL,
            self.equity / self.nav,
            self.cash / self.nav,
            self.leverage / self.max_leverage,
            (np.where(self.positions > 0)[0].shape[0] - np.where(self.positions < 0)[0].shape[0]) / self.positions.shape[0]
        ])
        # TODO: add proper feature engineering using above eval func
        features = self._evaluate_features()
        obs = np.concatenate([features, ptf_obs], axis=1)
        obs[np.bitwise_not(np.isfinite(obs))] = 0.
        return obs.reshape(self.shape)
    
    def _get_info(self):
        # return additional info about the agent in the environment
        return {
            "history":self.history,
            "feedback":self.feedback,
            "prices": self.df.iloc[:self.current_step].copy(deep=True),
            "net_worth": round(self.net_worths[-1], 2),
            "sharpe": round(self.sharpe(), 5)
        }
    
    def _done(self):
        """Check to see if iteration should stop"""
        should_stop = False
        if self.nav < self.initial_balance / 20:
            should_stop = True
        if self.current_step == self.max_steps - 1:
            should_stop = True
        return should_stop
    
    # TODO: add a render mode for system tracking (can also have human)
    def render(self, bad=None):
        net_worth = round(self.net_worths[-1], 2)
        initial_net_worth = round(self.net_worths[0], 2)
        profit_percent = round((net_worth - initial_net_worth) / initial_net_worth * 100, 2)
        print(f"\nCurrent Step {self.current_step}")
        print(f"Feedback: {self.feedback}")
        print(f'Net worth:' + str(net_worth) + ' | Profit: ' + str(profit_percent) + '%' + '| Sharpe Ratio:' + str(self.sharpe()))
        print(f'Last Reward: {round(self.last_reward, 4)} | Total trades: {self.num_trades}') # Current Portfolio Weights: {self.weights} and 


