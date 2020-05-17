import numpy as np
import pandas as pd


class RLCache(object):
    def __init__(self, no_cache_blocks, no_pages, base_reward,
                 learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.base_reward = base_reward
        self.cache_size = no_cache_blocks
        # self.cache = np.zeros(8, dtype=int)
        self.cache = pd.DataFrame(0, index=np.arange(no_cache_blocks), columns=['Pages'])
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.cntr = 0

        # qtable
        rows = np.math.factorial(no_pages) / (np.math.factorial(no_cache_blocks) * np.math.factorial((no_pages - no_cache_blocks)))
        cols = np.asarray(['State'])
        cols = np.append(cols, np.arange(1, no_pages+1))
        # print(cols[:10])
        cols = cols.flatten()
        # print(cols)
        self.qtable = pd.DataFrame(index=np.arange(int(rows)), columns=cols)
        self.qtable.iloc[:, 1:no_pages+1] = 0
        self.cost_list = []

    def get_cache(self):
        return self.cache

    def choose_action(self, page):
        if page in self.cache['Pages'].values:
            # print("HIT")
            return "HIT"
        else:
            # print("MISS")
            return "MISS"

    def get_reward(self, status):
        if status == 'HIT':
            reward = self.base_reward
        else:
            reward = -1

        return reward

    def step(self, action, page):
        if action == "HIT":
            reward = self.base_reward
        else:
            reward = -1

            state = self.cache['Pages'].values
            indices = self.cache.index[self.cache['Pages'] == 0].tolist()
            if len(indices) != 0:
                # print("Space available")
                self.cache['Pages'].iloc[indices[0]] = page
            else:
                # print("Space not available. Replacement needed")
                # select page to be removed
                replace_page = self.remove_page(state)
                # print("Remove Page: {}".format(replace_page))
                # update cache
                # print(self.cache.index[self.cache['Pages'] == replace_page].tolist())
                index = self.cache.index[self.cache['Pages'] == replace_page].tolist()[0]
                self.cache['Pages'].iloc[index] = page
        if self.cntr < self.cache_size:
            self.cntr += 1
        else:
            self.cntr = 0
        return self.cache['Pages'].values, reward


    def update_qtable(self, state, status, state_, reward, action):
        # print("update qtable")
        state.sort()

        state_str = str(state)
        state_str_ = str(state_)

        # Check old state in the Q-table
        if state_str in self.qtable['State'].values:
            index = self.qtable.index[self.qtable['State'] == state_str]
            index = index[0]
        else:
            indices = self.qtable.index[self.qtable['State'].isnull()]
            index = indices[0]
            # print("index: {}".format(index))
            self.qtable.iloc[index]['State'] = state_str

        # Check new state in the Q-table and get the max Q-value
        if state_str_ in self.qtable['State'].values:
            index_ = self.qtable.index[self.qtable['State'] == state_str_]
            index_ = index_[0]
        else:
            indices = self.qtable.index[self.qtable['State'].isnull()]
            index_ = indices[0]
            # print("index: {}".format(index_))
            self.qtable.iloc[index_]['State'] = state_str_

        # print("Next state Q-values: ", self.qtable.loc[index_, :].values[1:])
        max_val = self.qtable.loc[index_, :].values[1:].max()
        q_predict = self.qtable.iloc[index][action]
        if status == 'HIT':
            q_target = reward + self.gamma * max_val
            self.qtable.iloc[index][action] += self.lr * (q_target - q_predict)
        else:
            q_target = reward
            # print(state_str)
            for s in state:
                if s != 0:
                    # print(s, q_target, q_predict)
                    # print(self.qtable.iloc[index][s])
                    self.qtable.iloc[index][s] += self.lr * (q_target - q_predict)

    def remove_page(self, state):
        # print("remove page")
        state.sort()
        state_str = str(state)

        if np.random.uniform() < self.epsilon and (state_str in self.qtable['State'].values):
            index = self.qtable.index[self.qtable['State'] == state_str]
            # print("index: {}".format(index[0]))
            actions = self.qtable.iloc[index[0]][list(state)]
            remove_page = actions.values.argmin()
            # print("arg min: {}".format(remove_page))
            # print("page to be removed: ", state[remove_page])
            return state[remove_page]
        else:
            remove_page = np.random.choice(self.cache['Pages'].values)
            return remove_page

    def plot_reward(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_list)), self.cost_list)
        plt.ylabel('Reward')
        plt.xlabel('Training Steps')
        plt.show()
