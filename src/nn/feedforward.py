import numpy as np
from connlist import ConnList
from activations import get_act


class FeedForward():

    def __init__(self, configs):
        input_size = configs.input_size
        output_size = configs.output_size
        self.compiled = False
        self.nodeCount = input_size + output_size
        self.nodeLookup = {}
        self.connList = ConnList()
        self.inputs = np.zeros((input_size))
        self.outputs = np.zeros((input_size))
        self.input_size = input_size
        self.output_size = output_size
        self.values = None
        self.biases = []
        self.acts = []
        self.rawConnList = []
        for i in range(self.input_size):
            self.acts.append(get_act(0))
        for i in range(self.output_size):
            self.acts.append(get_act(1))

    def add_node(self, nodeName, act, bias):
        if (nodeName[0].isdigit()):
            if (not (nodeName in self.nodeLookup)):
                self.nodeLookup[nodeName] = self.nodeCount
                self.acts.append(get_act(act))
                self.biases.append(bias)
                self.nodeCount += 1

    def add_conn(self, source, target, weight):
        if (source[0] == 'i'):
            sourceId = int(source[1:])
        else:
            sourceId = self.nodeLookup[source]
        if (target[0] == 'o'):
            targetId = int(target[1:]) + self.input_size
        else:
            targetId = self.nodeLookup[target]
        self.connList.add_conn(sourceId, targetId, weight)
        self.rawConnList.append((source, target, weight))

    def topological_sort(self, node, nodeFlags, queue):
        if (nodeFlags[node] != 0):
            return
        indices = self.connList.get_all_conn_indices_source(node)
        if (indices is not None):
            targets = self.connList.connTarget[indices]
            nodeFlags[node] = 1
            for target in targets:
                self.topological_sort(target, nodeFlags, queue)
        nodeFlags[node] = 2
        queue.append(node)

    def add_finish(self):
        self.biases = np.array(self.biases, dtype=np.float16)
        self.connList.compile()
        self.nodeLookup = None

    def compile(self):
        self.connList.sort_by_source()
        nodeFlags = np.zeros(self.nodeCount, dtype=int)
        topologicalOrder = []
        for node in range(self.nodeCount):
            self.topological_sort(node, nodeFlags, topologicalOrder)
        topologicalOrder.reverse()
        self.connList.sort_by_target()
        sortIndex = []
        for node in topologicalOrder:
            indices = self.connList.get_all_conn_indices_target(node)
            if (indices is not None):
                sortIndex.extend(indices)
        self.connList.apply_indices(sortIndex)

    def step(self, inputs):
        self.values = np.zeros((self.nodeCount))
        self.values[0:self.input_size] = inputs
        #self.values[self.input_size + self.output_size:] = self.biases
        targetList = self.connList.connTarget
        sourceList = self.connList.connSource
        weightList = self.connList.connWeight
        connCount = self.connList.connCount
        if (connCount == 0):
            return self.values[self.input_size:self.input_size + self.output_size]
        prev = targetList[0]
        i = 0
        try:
            while (i < connCount):
                while ((i < connCount) & (prev == targetList[i])):
                    self.values[targetList[i]] += self.values[sourceList[i]] * \
                        weightList[i]
                    i += 1
                    if (i >= connCount):
                        break
                self.values[prev] = self.acts[prev](self.values[prev])
                if (i >= connCount):
                    break
                prev = targetList[i]
        except IndexError:
            print('step index error', i, connCount, len(targetList))
        return self.values[self.input_size:self.input_size + self.output_size]
