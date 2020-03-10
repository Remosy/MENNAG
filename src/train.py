import os
import sys
sys.path.append(os.getcwd() + '/src')
import argparse
import numpy as np
import gym
import multiprocessing as mp
from ea import EA
from nodes import Root
import time
import psutil
import random
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

def mpi_main(args, executor):
    ea = EA(1)
    ea.load_config(args.config)
    reseedPeriod = int(args.reseed)
    seed = 0
    for i in range(int(args.generation)):
        if ((reseedPeriod > 0) and (i % reseedPeriod == 0)):
            seed = random.randint(0, 2**32 - 1)
        print(seed)
        ea_time = time.time()
        pop = ea.ask()
        ea_time = time.time() - ea_time
        fitnesses = []
        workloads = []
        num_workers = int(args.num_workers) - 1
        for j in range(len(pop)):
            workloads.append(([pop[j]], args.task, seed))
        eval_time = time.time()
        results = executor.map(eval, workloads)
        for r in results:
            fitnesses.extend(r)
        eval_time = time.time() - eval_time
        ea.tell(fitnesses, args.task, seed)
        ea.write_history(args.output)
        print(i, ea.fitnesses[0], psutil.Process(os.getpid()).memory_info().rss, ea_time, eval_time, ea.pop[0].maxDepth)


def main(args):
    ea = EA(1)
    ea.load_config(args.config)
    for i in range(int(args.generation)):
        ea_time = time.time()
        pop = ea.ask()
        fitnesses = []
        ea_time = time.time() - ea_time
        num_workers = int(args.num_workers)
        if (num_workers == 1):
            fitnesses = eval((pop, args.task, 0))[0]
        else:
            batch_indices = np.linspace(0, len(pop), num_workers + 1).astype(int)
            batches = []
            for j in range(num_workers):
                batches.append((
                    pop[batch_indices[j]:batch_indices[j + 1]],
                    args.task,
                    j))
            eval_time = time.time()
            with mp.Pool(num_workers) as pool:
                results = pool.map(eval, batches)
            eval_time = time.time() - eval_time
            results.sort(key=lambda x: x[1])
            for r in results:
                fitnesses.extend(r[0])
        ea.tell(fitnesses, args.task, SEED)
        ea.write_history(args.output)
        print(i, ea.fitnesses[0], psutil.Process(os.getpid()).memory_info().rss, ea_time, eval_time)


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
    pop, task, seed = batches
    env = gym.make(task)
    env.seed(seed=seed)
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
            new_obs, reward, done, info = env.step(action)
            totalReward += reward
            if (max(abs((new_obs - obs))) < 1e-5):
                break
            obs = new_obs
        fitnesses.append(totalReward)
    env.close()
    return fitnesses

print(MPI.COMM_WORLD.Get_rank())
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', help='Task name')
    parser.add_argument('-c', '--config', help='Configuration file')
    parser.add_argument('-g', '--generation', help='Generation number')
    parser.add_argument('-n', '--num_workers', help='Number of cores', default=1)
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--reseed', default=-1)

    args = parser.parse_args()
    if args.mpi:
        with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:
                mpi_main(args, executor)
    else:
        main(args)
