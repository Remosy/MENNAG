import os
import sys
sys.path.append(os.getcwd() + '/src')
os.environ['MKL_NUM_THREADS'] = '1'
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

from eval.eval import eval, EvalSummary

DEBUG_FLAG = False

def mpi_main(args):
    DEBUG_FLAG = args.debug
    ea = EA(1)
    ea.load_config(args.config)
    reseedPeriod = int(args.reseed)
    taskNum = int(args.task_num)
    np.random.seed(1)
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
            workloads.append((pop[j], args.task, seed, args.debug))
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
        reducedResult = EvalSummary()
        reducedResult.reduce(results, 'pfit')
        ea.tell(reducedResult, args.task, seed)
        ea.write_history(args.output)
        #print(ea.fitnesses)
        print('iter: {0} fit: {1}, pfit:{7} Q: {2}, ea_time: {3}, prep_time: {4}, eval_time: {5}, max_depth:{6}'.format(
                i,
                ea.fitnesses[0],
                np.mean(reducedResult.get_res('Q')[0]),
                ea_time,
                prep_time,
                eval_time,
                ea.pop[0].maxDepth,
                np.mean(reducedResult.get_res('pfit')[0])
            ))

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
        print('iter: {0} fit: {1}, Q: {2}, ea_time: {3}, prep_time: {4}, eval_time: {5}, max_depth:{6}'.format(
            i, ea.fitnesses[0], np.mean(ea.Q), ea_time, prep_time, eval_time, ea.pop[0].maxDepth))



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
