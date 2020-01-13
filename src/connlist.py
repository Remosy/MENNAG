import numpy as np


class ConnList():

    def __init__(self):
        self.connSource = []
        self.connTarget = []
        self.connWeight = []
        self.connCount = 0

    def add_conn(self, source, target, weight):
        self.connSource.append(source)
        self.connTarget.append(target)
        self.connWeight.append(weight)

    def compile(self):
        self.connSource = np.asarray(self.connSource, dtype=int)
        self.connTarget = np.asarray(self.connTarget, dtype=int)
        self.connWeight = np.asarray(self.connWeight, dtype=float)
        self.connCount = len(self.connSource)

    def sort_by_source(self):
        sortedIndex = np.argsort(self.connSource)
        self.apply_indices(sortedIndex)

    def sort_by_target(self):
        sortedIndex = np.argsort(self.connTarget)
        self.apply_indices(sortedIndex)

    def get_all_conn_indices_source(self, node):
        left = np.searchsorted(self.connSource, node)
        if (left == self.connCount):
            return None
        if (self.connSource[left] != node):
            return None
        right = left
        while (self.connSource[left - 1] == node):
            left -= 1
        while (self.connSource[right] == node):
            if (right + 1 == self.connCount):
                right = self.connCount
                break
            right += 1
        return np.arange(left, right)

    def get_all_conn_indices_target(self, node):
        left = np.searchsorted(self.connTarget, node)
        if (left == self.connCount):
            return None
        if (self.connTarget[left] != node):
            return None
        right = left
        while (self.connTarget[left - 1] == node):
            left -= 1
        while (self.connTarget[right] == node):
            if (right + 1 == self.connCount):
                right = self.connCount
                break
            right += 1
        return np.arange(left, right)

    def apply_indices(self, sortedIndex):
        self.connSource = self.connSource[sortedIndex]
        self.connTarget = self.connTarget[sortedIndex]
        self.connWeight = self.connWeight[sortedIndex]
