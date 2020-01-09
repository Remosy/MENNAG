import random
import numpy as np
import configs


class Root():

    def __init__(self, dict=None):
        self.ID = ''
        self.depth = 0
        if (dict is None):
            dict = {
                'max_depth': 10,
                'input_size': 4,
                'output_size': 2,
                'feedforward': True,
                'forward_prob': 0.8,
                'weight_mean': 0,
                'weight_std': 1,
                'perturb_std': 0.2,
            }
        self.configs = configs.Configs(dict)

    def generate(self):
        self.child = Div(self)
        self.child.generate()

    def parse(self):
        self.child.parse()
        self.neuronSet = self.child.neuronSet
        self.connSet = self.child.connSet
        self.inSet = self.child.inSet
        self.outSet = self.child.outSet



class TreeNode():

    def __init__(self, parent):
        self.parent = parent
        self.ID = parent.ID
        self.depth = parent.depth
        self.configs = parent.configs
        self.child = [None, None, None]
        self.rule = -1
        self.reset()

    def reset(self):
        self.neuronSet = set()
        self.connSet = set()
        self.inSet = set()
        self.outSet = set()

    def mutate(self):
        return

    def generate(self):
        for c in self.child:
            if (c is not None):
                c.generate()
        return

    def parse(self):
        self.reset()
        for c in self.child:
            if (c is not None):
                c.parse()
        self.merge_from_children()
        return

    def update_depth(self):
        self.depth = self.parent.depth
        for c in self.child:
            if (c is not None):
                c.update_depth()
        return

    def max_depth_from_current(self):
        if (self.child[0] is not None):
            left_depth = self.child[0].max_depth_from_current()
        else:
            left_depth = self.depth
        if (self.child[1] is not None):
            right_depth = self.child[1].max_depth_from_current()
        else:
            right_depth = self.depth
        return max(left_depth, right_depth)

    def merge_from_children(self):
        for c in self.child:
            if (c is not None):
                self.neuronSet.update(c.neuronSet)
                self.connSet.update(c.connSet)
                self.outSet.update(c.outSet)
                self.inSet.update(c.inSet)


class Div(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)
        self.depth += 1

    def generate(self):
        number = random.random()
        if (number < 0.5):
            if (random.random() < 0.5):
                # Rule #2 DIV -> DIV CLONES CONNS
                self.child[0] = Div(self)
                self.child[0].ID += '0'
                self.child[1] = Clones(self, self.child[0])
                self.child[1].ID += '1'
                self.child[2] = Conns(self)
                self.rule = 2
            else:
                # Rule #3 DIV -> CLONES DIV CONNS
                self.child[1] = Div(self)
                self.child[1].ID += '1'
                self.child[0] = Clones(self, self.child[1])
                self.child[0].ID += '0'
                self.child[2] = Conns(self)
                self.rule = 3
        elif (number < 1 / (self.depth**2)):
            # Rule #0 DIV -> DIV DIV CONNS
            self.child[0] = Div(self)
            self.child[0].ID += '0'
            self.child[1] = Div(self)
            self.child[1].ID += '1'
            self.child[2] = Conns(self)
            self.rule = 0
        else:
            # Rule #1 DIV -> CELL CELL CONNS
            self.child[0] = Cell(self)
            self.child[0].ID += '0'
            self.child[1] = Cell(self)
            self.child[1].ID += '1'
            self.child[2] = Conns(self)
            self.rule = 1
        super().generate()

    def parse(self):
        self.reset()
        if (self.rule == 3):
            self.child[1].parse()
            self.child[0].parse()
            self.child[2].parse()
        else:
            self.child[0].parse()
            self.child[1].parse()
            self.child[2].parse()
        self.merge_from_children()

    def update_depth(self):
        self.depth = self.parent.depth + 1
        for c in self.child:
            if (c is not None):
                c.update_depth()
        return


class Clones(TreeNode):

    def __init__(self, parent, sibling):
        super().__init__(parent)
        self.sibling = sibling
        self.depth += 1

    def generate(self):
        number = random.random()
        if (number < 0.01):
            # rule #0 CLONES -> CLONES CLONE
            self.child[0] = Clone(self, self.sibling)
            self.child[0].ID += '0'
            self.child[1] = Clones(self, self.sibling)
            self.child[1].ID += '1'
            self.rule = 0
        else:
            # rule #1 CLONES -> CLONE
            self.child[0] = Clone(self, self.sibling)
            self.rule = 1
        super().generate()

    def parse(self):
        self.reset()
        if (self.rule == 0):
            self.child[0].parse()
            self.child[1].parse()
        else:
            self.child[0].parse()
        self.merge_from_children()

    def update_depth(self):
        self.depth = self.parent.depth + 1
        for c in self.child:
            if (c is not None):
                c.update_depth()
        return


class Clone(TreeNode):

    def __init__(self, parent, sibling):
        super().__init__(parent)
        self.sibling = sibling

    def generate(self):
        self.permi = np.random.rand(self.configs.input_size)
        self.permo = np.random.rand(self.configs.output_size)

    def parse(self):
        self.reset()
        sibDepth = self.sibling.depth
        argPermi = np.argsort(self.permi)
        argPermo = np.argsort(self.permo)
        for neuron in self.sibling.neuronSet:
            copy = self.ID + neuron[sibDepth:]
            self.neuronSet.add(copy)
        for conn in self.sibling.connSet:
            sourceCopy = self.ID + conn[0][sibDepth:]
            targetCopy = self.ID + conn[1][sibDepth:]
            self.connSet.add((sourceCopy, targetCopy, conn[2]))
        for inConn in self.sibling.inSet:
            sourceCopy = argPermi[inConn[0]]
            targetCopy = self.ID + inConn[1][sibDepth:]
            self.inSet.add((sourceCopy, targetCopy, inConn[2]))
        for outConn in self.sibling.outSet:
            sourceCopy = self.ID + outConn[0][sibDepth:]
            targetCopy = argPermo[outConn[1]]
            self.outSet.add((sourceCopy, targetCopy, outConn[2]))


class Conns(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        number = random.random()
        if (number < 0.5):
            # CONNS -> CONN CONNS
            self.child[0] = Conn(self)
            self.child[1] = Conns(self)
        else:
            # CONNS -> CONN
            self.child[0] = Conn(self)
        super().generate()


class Conn(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        max_depth = self.configs.max_depth
        self.sourceTail = bin(random.getrandbits(max_depth))[2:]
        self.targetTail = bin(random.getrandbits(max_depth))[2:]
        if (self.configs.feedforward):
            # feedforward connection
            self.sourceBase = self.ID + '0'
            self.targetBase = self.ID + '1'
        else:
            number = random.random()
            if (number < self.configs.forward_prob):
                self.sourceBase = self.ID + '0'
                self.targetBase = self.ID + '1'
            else:
                self.sourceBase = self.ID
                self.targetBase = self.ID

        mean = self.configs.weight_mean
        std = self.configs.weight_std
        self.weight = np.random.normal(mean, std)

    def parse(self):
        self.reset()
        d = self.max_depth_from_current()
        source = self.sourceBase + self.sourceTail
        source = source[:d]
        target = self.targetBase + self.targetTail
        target = target[:d]
        self.connSet.add((source, target, self.weight))

    def max_depth_from_current(self):
        p = self.parent
        while(isinstance(p, Conns)):
            p = p.parent
        return p.max_depth_from_current()


class Cell(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        self.child[0] = In(self)
        self.child[1] = Out(self)
        super().generate()

    def parse(self):
        self.reset()
        self.neuronSet.add(self.ID)
        self.child[0].parse()
        self.child[1].parse()
        self.merge_from_children()


class In(TreeNode):
    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        number = random.random()
        if (number < 0.5):
            # Rule #0 No connection
            self.rule = 0
        else:
            # Rule #1 In -> IO in?
            self.rule = 1
            self.source = random.choice(range(0, self.configs.input_size))
            mean = self.configs.weight_mean
            std = self.configs.weight_std
            self.weight = np.random.normal(mean, std)

    def parse(self):
        self.reset()
        if (self.rule == 1):
            self.inSet.add((self.source, self.ID, self.weight))


class Out(TreeNode):
    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        number = random.random()
        if (number < 0.5):
            # Rule #0 No connection
            self.rule = 0
        else:
            # Rule #1 Out -> IO out?
            self.rule = 1
            self.target = random.choice(range(0, self.configs.output_size))
            mean = self.configs.weight_mean
            std = self.configs.weight_std
            self.weight = np.random.normal(mean, std)

    def parse(self):
        self.reset()
        if (self.rule == 1):
            self.outSet.add((self.ID, self.target, self.weight))
