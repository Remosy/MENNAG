import os
import sys
import argparse
import numpy as np
import gym
from mpi4py import MPI
from ea import EA
from nodes import Root


def main(args):
    ea = EA()
    ea.load_config(args.config)
    for i in range(int(args.generation)):
        pop = ea.ask()
        if (args.num_workers == 1):
            fitnesses = eval(pop, args.task)
        ea.tell(fitnesses)
        print(max(fitnesses))


def eval(pop, task):
    env = gym.make(task)
    env.seed(seed=142857963)
    fitnesses = []
    for p in pop:
        nn = p.execute()
        obs = env.reset()
        done = False
        reward = 0
        while (not done):
            action = nn.step(obs)
            obs, reward, done, info = env.step(action)
        fitnesses.append(reward)
    env.close()
    return fitnesses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='Task name')
    parser.add_argument('-c', '--config', help='Configuration file')
    parser.add_argument('-g', '--generation', help='Generation number')
    parser.add_argument('-n', '--num_workers', help='Number of cores', default=1)
    args = parser.parse_args()

    main(args)
