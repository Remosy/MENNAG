import gym
from tasks.retina import Retina
from tasks.retinaNM import RetinaNM

def get_task(name):
    if (name == 'Retina'):
        return Retina()
    elif (name == 'RetinaNM'):
        return RetinaNM()
    else:
        return gym.make(name)
