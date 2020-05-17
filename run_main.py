import numpy as np
import pandas as pd
from RLCache import RLCache
from LRUCache import LRUCache
from FIFOCache import FIFOCache
from LFUCache import LFUCache
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == '__main__':
    sequence = [1, 4, 8, 5, 2, 7, 9, 7, 9, 1, 4, 10, 5, 6, 9, 7]
    # sequence = [7,5,1,2,5,3,5,4,2,3,5]
    # print("Sequence: {}".format(sequence))

    no_cache_blocks = 4
    no_pages = 10
    base_reward = 10

    random.seed(20)
    sequence = [random.randint(1, no_pages) for i in range(2000)]
    print("Length of Sequence: ", len(sequence))

    rlCache = RLCache(no_cache_blocks, no_pages, base_reward)
    rl_hits = 0
    lru_cache = LRUCache(no_cache_blocks)
    lru_hits = 0
    fifo_cache = FIFOCache(no_cache_blocks)
    fifo_hits = 0
    lfu_cache = LFUCache(sequence, no_pages, no_cache_blocks)
    lfu_hits = 0


    # print("Initial Cache State: \n {}".format(rlCache.get_cache()))
    # print("---------------------------------------------------------------")
    total_no_requests = len(sequence)
    # hits = 0

    t = 1
    hit_rate_t = []
    time = 500
    time = total_no_requests
    # time = 2000

    state = rlCache.cache['Pages'].values
    # print("Initial state: ", state)
    print("Processing Requests....")

    pbar = tqdm(total=time)
    i = 0

    while t <= time:
        # print("Page Word Address: {}".format(page))
        # page = random.choice(sequence)

        page = sequence[i]
        i += 1
        # if i >= len(sequence):
        #     i = 0

        cache_hit_rate = []

        # print("Page Word Address: {}".format(page))
        # -------------- Process RL --------------------
        # state = rlCache.cache['Pages'].values
        action = rlCache.choose_action(page)
        if action == "HIT":
            rl_hits += 1

        state_, reward = rlCache.step(action, page)

        if rlCache.cntr > rlCache.cache_size - 1:
            # rlCache.update_qtable(state, 'MISS', reward, action)
            rlCache.update_qtable(state, action, state_, reward, page)

        state = state_

        # ------------- Process for LRU return -------------
        val = lru_cache.get(page)
        if val == "HIT":
            lru_hits += 1

        # ------------- Process for FIFO return -------------
        val = fifo_cache.get(page)
        if val == "HIT":
            fifo_hits += 1

        # ------------- Process for LFU return -------------
        val = lfu_cache.get(page)
        if val == "HIT":
            lfu_hits += 1

        cache_hit_rate.append(rl_hits/t)
        cache_hit_rate.append(lru_hits/t)
        cache_hit_rate.append(fifo_hits/t)
        cache_hit_rate.append(lfu_hits/t)

        hit_rate_t.append(cache_hit_rate)

        t += 1
        pbar.update(1)

    # print(hit_rate_t[time-1])

    # plot the result over graph
    plt.style.use('seaborn-darkgrid')
    my_dpi = 96
    plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)

    x = [i for i in range(1,time+1)]

    cols = ['RLCache', 'LRUCache', 'FIFOCache', 'LFUCache']
    df = pd.DataFrame(hit_rate_t, columns=cols)
    color = ['red', 'blue', 'green', 'orange']

    # multiple line plot
    i = 0
    for column in df.columns:
        plt.plot(x, df[column], marker='', color=color[i], linewidth=1, alpha=0.4, label=column)
        i += 1

    # Add titles
    t = "Cache Hit-Rate ( with Cache Size: " + str(no_cache_blocks) + ", # of Pages: " + str(no_pages) + \
        ", Randomly Selected from I/p Sequence Length of " + str(total_no_requests) + ")"
    plt.title(t, fontsize=12, fontweight=0)
    plt.xlabel("Time Period")
    plt.ylabel("Cache Hit-Rate")
    plt.legend()
    plt.show()
