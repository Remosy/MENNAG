import random
import numpy as np
import configs


class Root():

    def __init__(self, dict=None):
        self.ID = ''
        self.depth = 0
        if (configs is None):
            dict = {
                'input_size': 4,
                'output_size': 2,
                'feedforward': True,
                'weight_mean': 0,
                'weight_std': 1,
                'perturb_std': 0.2,
            }
        self.configs = configs.Configs(dict)

    def generate(self):
        self.child = Div(self)
        self.child.generate()


class TreeNode():

    def __init__(self, parent):
        self.parent = parent
        self.ID = parent.ID
        self.depth = parent.depth
        self.configs = parent.configs
        reset()

    def reset(self):
        self.child = [None, None, None]
        self.neuronList = []
        self.connList = []
        self.inList = []
        self.outList = []
        self.rule = -1

    def mutate(self, configs):
        return

    def generate(self):
        for c in self.child:
            if (c is not None):
                c.generate()
        return

    def compile(self):
        return

    def update(self):
        self.depth = self.parent.depth
        for c in self.child:
            if (c is not None):
                c.update()
        return

    def max_depth_from_current(self):
        if (self.child[0] is not None):
            left_depth = max_depth_from_current(self.child[0])
        else:
            left_depth = self.depth
        if (self.child[1] is not None):
            right_depth = max_depth_from_current(self.child[1])
        else:
            right_depth = self.depth
        return max(left_depth, right_depth)

    def merge_from_children(self):
        for c in self.child:
            if (c is not None):
                self.neuronList += c.neuronList
                self.connList += c.connList
                self.outList += c.outList
                self.inList += c.inList


class Div(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)
        self.depth += 1

    def generate(self):
        number = random.random()
        if (number < 0.01):
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
                self.child[0] = Clones(self, self.child[1])
                self.child[0].ID += '0'
                self.child[1] = Div(self)
                self.child[1].ID += '1'
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
            self.child[1] = Cell(self)
            self.child[2] = Conns(self)
            self.rule = 1
        super().generate()

    def compile(self):
        self.reset()
        if (self.rule == 3):
            self.child[1].compile()
            self.child[0].compile()
            self.child[2].compile()
        else:
            self.child[0].compile()
            self.child[1].compile()
            self.child[2].compile()
        self.merge_from_children()

    def update_depth(self):
        self.depth = self.parent.depth + 1


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
            self.child[0] = Clone(self)
            self.rule = 1
        super().generate()

    def compile(self):
        self.reset()
        if (self.rule == 0):
            self.child[0].compile()
            self.child[1].compile()
        else:
            self.child[0].compile()
        merge_from_children()

    def update_depth(self):
        self.depth = self.parent.depth + 1


class Clone(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        self.permi = np.random((self.configs['input_size'],))
        self.permo = np.random((self.configs['output_size'],))

    def compile(self):


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


class Cell(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        self.child[0] = In(self)
        self.child[1] = Out(self)
        super.generate()

    def compile(self):
        self.reset()
        self.neuronList = [self.ID]
        self.child[0].compile()
        self.child[1].compile()
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

    def compile(self):
        self.reset()
        if (self.rule == 1):
            self.inList.append([self.source, self.ID, self.weight])


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
            self.target = random.choice(range(0, self.configs.input_size))
            mean = self.configs.weight_mean
            std = self.configs.weight_std
            self.weight = np.random.normal(mean, std)

    def compile(self):
        self.reset()
        if (self.rule == 1):
            self.inList.append([self.ID, self.target, self.weight])
