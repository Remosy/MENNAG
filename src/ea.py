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
        keys = np.flip(np.argsort(self.fitnesses)).tolist()
        pop = [self.pop[i] for i in keys]
        self.pop = pop

    def rank(self):
        rank = np.argsort(self.fitnesses)
        return (rank + 1) / sum(rank)

    def reproduce(self):
        newPop = []
        popSize = self.config.pop_size
        elitismRatio = self.config.elitism_ratio
        p = np.flip(np.array(range(self.config.pop_size)))
        p = (p + 1) / sum(p + 1)
        while (len(newPop) < round(popSize * self.config.cross_rate)):
            parent1 = np.random.choice(self.pop, p=p)
            parent2 = np.random.choice(self.pop, p=p)
            if (parent1 != parent2):
                offspring = parent1.cross_with(parent2)
                offspring.compile()
                newPop.append(offspring)
        for i in range(round(popSize * elitismRatio)):
            newPop.append(self.pop[i].deepcopy())
        while(len(newPop) < popSize):
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
