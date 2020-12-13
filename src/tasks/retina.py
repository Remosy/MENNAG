import numpy as np
from gym.utils import seeding

def bin_to_int(bits):
    l = len(bits)
    result = 0
    for i in range(l):
        result += bits[i] << (l - i - 1)
    return result

class Retina:

    def __init__(self):
        self.seed()
        fourBits = list(range(16))
        self.pl = self.np_random.choice(fourBits, 8, replace=False)
        self.pr = self.np_random.choice(fourBits, 8, replace=False)
        self.current = 0

    def reset(self):
        self.current = 0
        fourBits = list(range(16))
        self.pl = self.np_random.choice(fourBits, 8, replace=False)
        self.pr = self.np_random.choice(fourBits, 8, replace=False)
        return self.get_state()

    def step(self, action):
        s = self.get_state()
        l = s[0:4]
        r = s[4:8]
        lc = False
        lNum = bin_to_int(l)
        for p in self.pl:
            if (p == lNum):
                lc = True
                break
        rc = False
        rNum = bin_to_int(r)
        for p in self.pr:
            if (p == rNum):
                rc = True
                break
        self.current += 1
        if ((lc & rc) ^ action):
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
