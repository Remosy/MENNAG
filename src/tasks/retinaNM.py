import numpy as np
from gym.utils import seeding

def bin_to_int(bits):
    l = len(bits)
    result = 0
    for i in range(l):
        result += bits[i] << (l - i - 1)
    return result

class RetinaNM:

    def __init__(self):
        self.seed()
        bits = list(range(256))
        self.p = self.np_random.choice(bits, 64)
        self.current = 0

    def reset(self):
        bits = list(range(256))
        self.p = self.np_random.choice(bits, 64)
        self.current = 0
        return self.get_state()

    def step(self, action):
        s = self.get_state()
        n = bin_to_int(s)
        r = False
        for p in self.p:
            if (p == n):
                r = True
                break
        self.current += 1
        if (action ^ r):
            reward = 1 / 256
        else:
            reward = 0
        if (self.current == 256):
            done = True
        else:
            done = False

        return self.get_state(), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def get_state(self):
        s = np.binary_repr(self.current, width = 8)
        a = np.zeros(8, dtype=int)
        for i in range(8):
            a[i] = int(s[i])
        return a

    def close(self):
        return
