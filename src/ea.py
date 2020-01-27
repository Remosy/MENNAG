import numpy as np
from configs import Configs
from nodes import Root

class EA():

    def __init__(self, config):
        self.pop = []
        self.fitnesses = np.zeros((config.pop_size))
        self.newPop = []

    def ask(self):
        if (len(self.pop) == 0):
            for i in range(self.config.pop_size):
                newInd = Root(config=self.config)
                newInd.generate()
                newInd.compile()
                self.pop.append(newInd)
            return self.pop
        else:
            return self.pop

    def tell(self, fitnesses):
        self.fitnesses = fitnesses
        
