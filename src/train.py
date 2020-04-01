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
import gc
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from visualizer.individualvisualizer import IndividualVisualizer

def mpi_main(args):
    ea = EA(1)
    ea.load_config(args.config)
    reseedPeriod = int(args.reseed)
    taskNum = int(args.task_num)
    np.random.seed(0)
    seed = np.random.randint(0, 2**32 - 1, size=(taskNum), dtype=np.uint32)
    seed = seed.tolist()
    print(seed)
    for i in range(int(args.generation)):
        if ((reseedPeriod > 0) and (i % reseedPeriod == 0)):
            for j in range(taskNum):
                seed[j] = random.randint(0, 2**32 - 1)
        ea_time = time.time()
        pop = ea.ask()
        ea_time = time.time() - ea_time
        fitnesses = []
        workloads = []
        num_workers = int(args.num_workers) - 1
        gc.collect()
        prep_time = time.time()
        for j in range(len(pop)):
            workloads.append((pop[j].execute(), args.task, seed))
        prep_time = time.time() - prep_time
        eval_time = time.time()
        success = False
        while (success is False):
            try:
                with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
                    if executor is not None:
                        results = executor.map(eval, workloads)
                        success = True
            except OverflowError:
                success = False
        eval_time = time.time() - eval_time
        ea.tell(results, args.task, seed)
        ea.write_history(args.output)
        print(i, ea.fitnesses[0], psutil.Process(os.getpid()).memory_info().rss, ea_time, prep_time, eval_time, ea.pop[0].maxDepth)
    best = ea.pop[0]
    eval((best.execute(), args.task, seed), debug = True)


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
        ea.tell(resultss, args.task, SEED)
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

def eval(batches, debug=False, render=False):
    try:
        nn, task, seeds = batches
        env = gym.make(task)
        fitnesses = []
        nn.compile()
        for seed in seeds:
            env.seed(seed=seed)
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
                if (debug):
                    print(action, obs)
                new_obs, reward, done, info = env.step(action)
                totalReward += reward
                obs = new_obs
                if render:
                    env.render()
            fitnesses.append(totalReward)
            env.close()
        fitnesses = np.array(fitnesses, dtype=np.float16)
    except OverflowError:
        print("OVERFLOW IN EVAL")
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
    parser.add_argument('--task_num', default=1)

    args = parser.parse_args()
    if args.mpi:
        mpi_main(args)
    else:
        main(args)
