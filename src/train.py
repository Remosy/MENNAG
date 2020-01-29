import os
import sys
import argparse
import numpy as np
import gym

from ea import EA
from nodes import Root


def main(args):
    ea = EA()
    ea.load_config(args.config)
    for i in range(args.generation):
        pop = ea.ask()
        if (args.num_workers == 1):
            fitnesses = eval(pop, args.task)
        ea.tell(fitnesses)

def eval(pop, task):
    env = gym.make(task)
    env.seed(seed=142857963)
    for p in pop:
        env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='Task name')
    parser.add_argument('-c', '--config', help='Configuration file')
    parser.add_argument('-g', '--generation', help='Generation number')
    parser.add_argument('-n', '--num_workers', help='Number of cores', default=1)
    args = parser.parse_args()

    main(args)
