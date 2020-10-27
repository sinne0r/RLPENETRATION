#################
# импорты
import gym, bitmex
from gym import spaces
from gym.utils import seeding   
import random, json
import pandas as pd
import numpy as np
#################
class SandboxSnd(gym.Env):
    metadata = {'render.modes': ['human']}
#################
    def __init__(self):
        super(SandboxSnd, self).__init__()
        #################
        global MAX_ACCOUNT_BALANCE 
        global MAX_NUM_SHARES 
        global MAX_SHARE_PRICE 
        global MAX_STEPS
        global INITIAL_ACCOUNT_BALANCE 
        global BIN_SIZE 
        global SYMBOL 
        global OBSERV_WIN 
        # константы
        MAX_ACCOUNT_BALANCE = 1
        MAX_NUM_SHARES = 100000000000000
        MAX_SHARE_PRICE = 1/13000
        MAX_STEPS = 2000
        INITIAL_ACCOUNT_BALANCE = 0.01
        BIN_SIZE = '5m'
        SYMBOL = 'XBTUSD'
        OBSERV_WIN = 5
        #################
        self.df = self._get_df(BIN_SIZE,SYMBOL,MAX_STEPS)
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = self.action_space = spaces.Discrete(2) # купить продать
        # или  self.action_space = spaces.Box(low = np.array([0, 0]), high = np.array([3, 1]), dtype=np.float16) для случая действие + количество
        
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)
#################
    def _get_df(self, binSize, symbol, MAX_STEPS):
        client = bitmex.bitmex(test=True)
        out = []
        for _ in range(0, MAX_STEPS, 1000):
            page = client.Trade.Trade_getBucketed(
                    binSize = binSize,
                    symbol = symbol,
                    count = 1000,
                    start = _ ,
                    reverse=True
                ).result()[0]
            out.extend(page)
        out.reverse()
        df = pd.DataFrame(out)
        df['volume'] = df['volume'].astype('double')
        df = df.fillna(df.median())
        return df
 ###################
    def _get_stat(self):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        return profit
 ###################
    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            (1/self.df.loc[self.current_step: self.current_step +
                        OBSERV_WIN, 'open'].values) / MAX_SHARE_PRICE,
            (1/self.df.loc[self.current_step: self.current_step +
                        OBSERV_WIN, 'high'].values) / MAX_SHARE_PRICE,
            (1/self.df.loc[self.current_step: self.current_step +
                        OBSERV_WIN, 'low'].values) / MAX_SHARE_PRICE,
            (1/self.df.loc[self.current_step: self.current_step +
                        OBSERV_WIN, 'close'].values) / MAX_SHARE_PRICE,
            (self.df.loc[self.current_step: self.current_step +
                        OBSERV_WIN, 'volume'].values) / MAX_NUM_SHARES])
        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs
#######################
    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = 1/random.uniform(self.df.loc[self.current_step,"open"],
                                         self.df.loc[self.current_step, "close"])
        action_type = action # когда просто купить или продать 10 контрактов
        # action_type = action[0] когда действие плюс кол-во
        # amount = action[1]
        
        total_possible = int(self.balance / current_price)
        shares_val = 10 #int(total_possible * amount)
        # фиксированно заключаем транзаакцию на 10 контрактов
        # все что ниже нужно для модели где число контрактов является action
        if action_type < 1:
            # Buy amount % of balance in shares
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_val * current_price
            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_val)
            self.shares_held += shares_val

        elif action_type < 2:
            # Sell amount % of shares held
            self.balance += shares_val * current_price
            self.shares_held -= shares_val
            self.total_shares_sold += shares_val
            self.total_sales_value += shares_val * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0
#################
    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > MAX_STEPS-OBSERV_WIN-1:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.net_worth * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}
#################
    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, MAX_STEPS-OBSERV_WIN-1)
        return self._next_observation()
#################
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        #print(f'Step: {self.current_step}')
        #print(f'Balance: {self.balance}')
        #print(
         #   f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        #print(
          #  f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        #print(
         #   f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
