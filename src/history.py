import os
import pickle
from pickle import UnpicklingError

class History():

    def __init__(self):
        self.store = []

    def __getitem__(self, key):
        return self.store[key]

    def load(self, filename):
        infile = open(filename, 'rb')
        while 1:
            try:
                item = pickle.load(infile)
                self.store.append(item)
            except (EOFError, UnpicklingError):
                break
        infile.close()
