import os
import sys
import argparse
import numpy as np
import gym
import multiprocessing as mp
from ea import EA
from nodes import Root


def main(args):
    ea = EA()
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
        ea.tell(fitnesses)
        print(max(fitnesses))


def eval(batches):
    pop, task, batch_num = batches
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
    return (fitnesses, batch_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='Task name')
    parser.add_argument('-c', '--config', help='Configuration file')
    parser.add_argument('-g', '--generation', help='Generation number')
    parser.add_argument('-n', '--num_workers', help='Number of cores', default=1)
    args = parser.parse_args()

    main(args)
