import numpy as np


def linear(x):
    return x
    

def sigmoid(x):
    return (np.tanh(x / 2.0) + 1.0) / 2.0


def get_act(num):
    if (num == 0):
        return linear
    if (num == 1):
        return sigmoid
