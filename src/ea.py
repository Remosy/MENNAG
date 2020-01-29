import numpy as np
from configs import Configs
from nodes import Root
import json


class EA():

    def __init__(self):
        self.pop = []

    def ask(self):
        if (len(self.pop) == 0):
            for i in range(self.config.pop_size):
                newInd = Root(config=self.config)
                newInd.generate()
                newInd.compile()
                self.pop.append(newInd)
            return self.pop
        else:
            self.reproduce()
            return self.pop

    def tell(self, fitnesses):
        self.fitnesses = fitnesses
        keys = np.argsort(self.fitnesses).reverse()
        self.pop = self.pop[keys]

    def rank(self):
        rank = np.argsort(self.fitnesses)
        return (rank + 1) / sum(rank)

    def reproduce(self):
        newPop = []
        popSize = self.configs.pop_size
        elitismRatio = self.config.elitism_ratio
        p = np.array(range(self.config.pop_size)).reverse()
        p = (p + 1) / sum(p + 1)
        while (len(self.newPop) < round(popSize * self.config.cross_rate)):
            parent1 = np.random.choice(self.pop, p=p)
            parent2 = np.random.choice(self.pop, p=p)
            if (parent1 != parent2):
                offspring = parent1.cross_with(parent2)
                offspring.compile()
                newPop.append()
        newPop.extend(self.pop[:round(popSize * elitismRatio)])
        while(len(self.newPop) < popSize):
            offspring = np.random.choice(self.pop, p=p).deepcopy()
            offspring.mutate()
            offspring.compile()
            newPop.append(offspring)
        self.pop = newPop

    def load_config(self, filename):
        with open(filename) as configFile:
            configDict = json.load(configFile)
        self.config = Configs(configDict)
        self.fitnesses = np.zeros((self.config.pop_size))
