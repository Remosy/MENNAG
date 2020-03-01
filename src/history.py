import os
import pickle
from pickle import UnpicklingError

class History():

    def __init__(self):
        self.tasks = []
        self.seeds = []
        self.bests = []
        self.fitnesses = []
        self.bestFit = []

    def load(self, filename):
        infile = open(filename, 'rb')
        while 1:
            try:
                item = pickle.load(infile)
                self.tasks.append(item.task)
                self.seeds.append(item.seed)
                self.bests.append(item.best)
                self.fitnesses.append(item.fitness)
            except (EOFError, UnpicklingError):
                break
        infile.close()

class HistItem():

    def __init__(self, task, seed, best, fitness):
        self.task = task
        self.seed = seed
        best.detach()
        self.best = best
        self.fitness = fitness
