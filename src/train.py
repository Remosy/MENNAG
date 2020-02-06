import os
import sys
import argparse
import numpy as np
import gym
import multiprocessing as mp
from ea import EA
from nodes import Root
import time

SEED = 142857369

def main(args):
    ea = EA(1)
    ea.load_config(args.config)
    for i in range(int(args.generation)):
        pop = ea.ask()
        fitnesses = []
        num_workers = int(args.num_workers)
        if (num_workers == 1):
            fitnesses = eval((pop, args.task, 0))[0]
        else:
            batch_indices = np.linspace(0, len(pop), num_workers + 1).astype(int)
            batches = []
            for i in range(num_workers):
                batches.append((
                    pop[batch_indices[i]:batch_indices[i + 1]],
                    args.task,
                    i))
            with mp.Pool(num_workers) as pool:
                results = pool.map(eval, batches)
            results.sort(key=lambda x: x[1])
            for r in results:
                fitnesses.extend(r[0])
        ea.tell(fitnesses, args.task, SEED)
        print(ea.fitnesses[0], np.mean(ea.fitnesses))

    ea.write_history(args.output)

def simulate(ind, task):
    env = gym.make(task)
    env.seed(seed=SEED)
    nn = ind.execute()
    obs = env.reset()
    done = False
    totalReward = 0
    while (not done):
        if (task == 'CartPole-v1'):
            action = nn.step(obs)[0]
            if (action > 0):
                action = 1
            else:
                action = 0
        else:
            action = nn.step(obs) * env.action_space.high
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()

def eval(batches):
    pop, task, batch_num = batches
    env = gym.make(task)
    env.seed(seed=SEED)
    fitnesses = []
    for ind in pop:
        nn = ind.execute()
        obs = env.reset()
        done = False
        totalReward = 0
        while (not done):
            if (task == 'CartPole-v1'):
                action = nn.step(obs)[0]
                if (action > 0):
                    action = 1
                else:
                    action = 0
            else:
                action = nn.step(obs) * env.action_space.high
            obs, reward, done, info = env.step(action)
            totalReward += reward
        fitnesses.append(totalReward)
    env.close()
    return (fitnesses, batch_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='Task name')
    parser.add_argument('-c', '--config', help='Configuration file')
    parser.add_argument('-g', '--generation', help='Generation number')
    parser.add_argument('-n', '--num_workers', help='Number of cores', default=1)
    parser.add_argument('-o', '--output', help='output file')
    args = parser.parse_args()

    main(args)
