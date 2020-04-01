import numpy as np
import math

def linear(x):
    return x


def sigmoid(x):
    x = np.clip(x, -100, 100)
    return (1 / (1 + math.exp(-3 * x)) - 0.5) * 2


def get_act(num):
    if (num == 0):
        return linear
    if (num == 1):
        return sigmoid
