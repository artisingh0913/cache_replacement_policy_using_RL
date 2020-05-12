import numpy as np
import pandas as pd
from RLCache import RLCache


if __name__ == '__main__':
    sequence = [1, 4, 8, 5, 2, 7, 9, 7, 9, 1, 4, 10, 5, 6, 9, 7]
    # sequence = [7,5,1,2,5,3,5,4,2,3,5]
    # print("Sequence: {}".format(sequence))
    no_cache_blocks = 4
    no_pages = 10
    base_reward = 10
    rlCache = RLCache(no_cache_blocks, no_pages, base_reward)
    print("Initial Cache State: \n {}".format(rlCache.get_cache()))
    print("---------------------------------------------------------------")
    total_no_requests = len(sequence)
    hits = 0

    for t, page in enumerate(sequence):
        print("Page Word Address: {}".format(page))
        state = rlCache.cache['Pages'].values
        action = page

        if page in rlCache.cache['Pages'].values:
            print("HIT")
            hits += 1
            reward = rlCache.get_reward('HIT')
            if t > no_cache_blocks-1:
                rlCache.update_qtable(state, 'HIT', reward, action)
        else:
            print("MISS")
            reward = rlCache.get_reward('MISS')
            indices = rlCache.cache.index[rlCache.cache['Pages'] == 0].tolist()
            if len(indices) != 0:
                print("Space available")
                rlCache.cache['Pages'].iloc[indices[0]] = page
            else:
                if t > no_cache_blocks-1:
                    rlCache.update_qtable(state, 'MISS', reward, action)
                print("Space not available. Replacement needed")
                # select page to be removed
                replace_page = rlCache.remove_page(state)
                print("Remove Page: {}".format(replace_page))
                # update cache
                index = rlCache.cache.index[rlCache.cache['Pages'] == replace_page].tolist()[0]
                rlCache.cache['Pages'].iloc[index] = page

            print("Current Cache State: \n {}".format(rlCache.cache))

        # if t % 4 == 0 and t != 0:
        # if t > 3:
        #     rlCache.update_qtable(state, reward, action)
        print("---------------------------------------------------------------")

    print("Hit Rate: {}".format(hits/total_no_requests))