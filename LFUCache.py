import numpy as np


class LFUCache:

    def __init__(self, sequence, no_pages, no_cache_blocks):
        self.sequence = sequence
        self.no_cache_blocks = no_cache_blocks
        self.no_pages = no_pages
        self.frame = np.zeros(self.no_cache_blocks, dtype='int32')
        self.freq = np.zeros(self.no_cache_blocks, dtype='int32')
        self.count = np.zeros(self.no_pages + 1, dtype='int32')
        self.cntr = 0
        self.hits = 0

    def get_hits(self):
        p = len(self.sequence)
        f = self.no_cache_blocks
        pages = self.sequence
        frame = np.zeros(self.no_cache_blocks, dtype='int32')
        pageHit = 0
        freq = np.zeros(self.no_cache_blocks, dtype='int32')
        count = np.zeros(self.no_pages + 1, dtype='int32')
        for i in range(p):
            flag = True
            page = pages[i]

            for j in range(f):
                if frame[j] == page:
                    flag = False
                    pageHit += 1
                    count[page] += 1
                    break

            if flag:
                if i >= 3:
                    for j in range(f):
                        num = frame[j]
                        freq[j] = count[num]
                    mini = freq[0]
                    for j in range(f):
                        if freq[j] < mini:
                            mini = freq[j]

                    for j in range(f):
                        if freq[j] == mini:
                            count[page] += 1
                            frame[j] = page
                            break
                else:
                    frame[i] = page
                    count[page] += 1

            # print("frame : ")
            # for k in range(f):
            #     print(frame[k], end=' ')
            # print()

        # print("hits: {}".format(pageHit))
        return pageHit

    def get(self, page):

        flag = True

        for j in range(self.no_cache_blocks):
            # HIT
            if self.frame[j] == page:
                flag = False
                self.hits += 1
                self.count[page] += 1
                return "HIT"
        # MISS
        if flag:
            if self.cntr >= 3:
                for j in range(self.no_cache_blocks):
                    num = self.frame[j]
                    self.freq[j] = self.count[num]
                mini = self.freq[0]
                for j in range(self.no_cache_blocks):
                    if self.freq[j] < mini:
                        mini = self.freq[j]

                for j in range(self.no_cache_blocks):
                    if self.freq[j] == mini:
                        self.count[page] += 1
                        self.frame[j] = page
                        break
            else:
                self.frame[self.cntr] = page
                self.count[page] += 1

        if self.cntr < 3:
            self.cntr += 1

        return "MISS"