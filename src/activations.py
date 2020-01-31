import numpy as np
import math

def linear(x):
    return x


def sigmoid(x):
    return (1 / (1 + math.exp(-x)) - 0.5) * 2


def get_act(num):
    if (num == 0):
        return linear
    if (num == 1):
        return sigmoid
