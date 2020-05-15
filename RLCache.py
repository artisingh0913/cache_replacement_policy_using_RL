import numpy as np
import pandas as pd


class RLCache(object):
    def __init__(self, no_cache_blocks, no_pages, base_reward):
        self.base_reward = base_reward
        # self.cache = np.zeros(8, dtype=int)
        self.cache = pd.DataFrame(0, index=np.arange(no_cache_blocks), columns=['Pages'])

        # qtable
        rows = np.math.factorial(no_pages) / (np.math.factorial(no_cache_blocks) * np.math.factorial((no_pages - no_cache_blocks)))
        cols = np.asarray(['State'])
        cols = np.append(cols, np.arange(1, no_pages+1))
        # print(cols[:10])
        cols = cols.flatten()
        print(cols)
        self.qtable = pd.DataFrame(index=np.arange(int(rows)), columns=cols)
        self.qtable.iloc[:, 1:no_pages+1] = 0
        self.cost_list = []

    def get_cache(self):
        return self.cache

    def get_reward(self, status):
        if status == 'HIT':
            reward = self.base_reward
        else:
            reward = -1

        return reward

    def update_qtable(self, state, status, reward, action):
        # print("update qtable")
        state.sort()
        # next_state =
        state_str = str(state)

        if state_str in self.qtable['State'].values:
            index = self.qtable.index[self.qtable['State'] == state_str]
            index = index[0]
        else:
            indices = self.qtable.index[self.qtable['State'].isnull()]
            index = indices[0]
            # print("index: {}".format(index))
            self.qtable.iloc[index]['State'] = state_str

        ### TODO: should the q-update be not only for the state received.
        if status == 'HIT':
            # self.qtable.iloc[:, action] += reward
            self.qtable.iloc[index][action] += reward
        else:
            for s in state:
                # self.qtable.iloc[:, s] += reward
                self.qtable.iloc[index][s] += reward

        self.cost_list.append(reward)

        # print(self.qtable[self.qtable['State'] == state_str])

    def remove_page(self, state):
        # print("remove page")
        state.sort()
        state_str = str(state)
        index = self.qtable.index[self.qtable['State'] == state_str]
        # print("index: {}".format(index[0]))
        actions = self.qtable.iloc[index[0]][list(state)]
        remove_page = actions.values.argmin()
        # print("arg min: {}".format(remove_page))
        return state[remove_page]

    def plot_reward(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_list)), self.cost_list)
        plt.ylabel('Reward')
        plt.xlabel('Training Steps')
        plt.show()
