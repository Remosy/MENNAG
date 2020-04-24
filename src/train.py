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
from tasks.task import get_task
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
from analyzer import Q

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
            workloads.append((pop[j], args.task, seed))
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
        print(i, ea.fitnesses[0], np.mean(ea.Q), ea_time, prep_time, eval_time, ea.pop[0].maxDepth)

def main(args):
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
            workloads.append((pop[j], args.task, seed))
        prep_time = time.time() - prep_time
        eval_time = time.time()
        if (num_workers > 1):
            with mp.Pool(num_workers) as pool:
                results = pool.map(eval, workloads)
        else:
            results = []
            for w in workloads:
                results.append(eval(w))
        eval_time = time.time() - eval_time
        ea.tell(results, args.task, seed)
        ea.write_history(args.output)
        print(i, ea.fitnesses[0], np.mean(ea.Q), ea_time, prep_time, eval_time, ea.pop[0].maxDepth)

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
        pop, task, seeds = batches
        env = get_task(task)
        fitnesses = []
        pop.compile()
        nn = pop.execute()
        nn.compile()
        for seed in seeds:
            env.seed(seed=seed)
            obs = env.reset()
            done = False
            totalReward = 0
            while (not done):
                if (task in ('CartPole-v1', 'Retina', 'RetinaNM')):
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
                if (task == 'BipedalWalker-v3'):
                    if (max(abs(obs - new_obs)) < 1e-5):
                        done = True
                obs = new_obs
                if render:
                    env.render()
            fitnesses.append(totalReward)
            env.close()
        fitnesses = np.array(fitnesses)
        result = Q(nn)
        Q_value, groups = result
    except OverflowError:
        print("OVERFLOW IN EVAL")
    return (pop, fitnesses, Q_value)

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
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    print(args.mpi)
    if args.mpi:
        mpi_main(args)
    else:
        main(args)
