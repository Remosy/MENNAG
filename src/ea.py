import numpy as np
from configs import Configs
from nodes import Root
import json
import pickle


class EA():

    def __init__(self, taskNum):
        self.pop = []
        self.taskNum = taskNum
        self.writeInit = False

    def ask(self):
        if (len(self.pop) == 0):
            for i in range(self.config.pop_size):
                newInd = Root(config=self.config)
                newInd.generate()
                #newInd.compile()
                self.pop.append(newInd)
            return self.pop
        else:
            self.reproduce()
            return self.pop

    def tell(self, results, task, seed):
        self.results = results
        self.pop = results.get_res('pop')
        self.fitnesses = results.get_metric()
        results.summarise()

    def rank(self):
        rank = np.argsort(self.fitnesses)
        return (rank + 1) / sum(rank)

    def reproduce(self):
        newPop = []
        popSize = self.config.pop_size
        elitismRatio = self.config.elitism_ratio
        p = np.flip(np.array(range(self.config.pop_size))) + 1
        p = p / sum(p)
        while (len(newPop) < round(popSize * self.config.cross_rate)):
            parent1 = np.random.choice(self.pop, p=p)
            parent2 = np.random.choice(self.pop, p=p)
            if (parent1 != parent2):
                offspring = parent1.cross_with(parent2)
                #offspring.compile()
                newPop.append(offspring)
        for i in range(round(popSize * elitismRatio)):
            offspring = self.pop[i].deepcopy()
            #offspring.compile()
            newPop.append(offspring)
        while(len(newPop) < popSize):
            offspring = np.random.choice(self.pop, p=p).deepcopy()
            offspring.mutate()
            #offspring.compile()
            newPop.append(offspring)
        self.pop = newPop

    def load_config(self, filename):
        with open(filename) as configFile:
            configDict = json.load(configFile)
        self.config = Configs(configDict)
        self.fitnesses = np.zeros((self.config.pop_size))

    def write_history(self, filename):
        if (not self.writeInit):
            outfile = open(filename, 'wb')
            self.writeInit = True
        else:
            outfile = open(filename, 'ab')
        pickle.dump(self.results, outfile)
        outfile.close()
