from queue import Queue

class FIFOCache:

    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = set()
        self.index = Queue()
        self.capacity = capacity
        self.hit = 0

    def get(self, page):

        if page not in self.cache:
            if len(self.cache) < self.capacity:
                self.cache.add(page)
                self.index.put(page)
            else:
                val = self.index.queue[0]
                self.index.get()
                self.cache.remove(val)
                self.cache.add(page)
                self.index.put(page)
            return "MISS"
        else:
            self.hit += 1
            return "HIT"