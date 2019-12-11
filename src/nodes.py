import random
import numpy as np

class TreeNode():

    def __init__(self, parent):
        self.parent = parent
        self.ID = parent.ID
        self.depth = parent.depth
        self.child = [None, None, None]
        self.neuronList = []
        self.connList = []
        self.rule = -1
        self.configs = parent.configs

    def mutate(self, configs):
        return

    def generate(self):
        for c in self.child:
            if (c is not None):
                c.generate()
        return

    def compile(self):
        return

    def merge_from_children(self):
        if hasattr(self, 'neuronList'):
            for c in self.child:
                if (c is not None):
                    self.neuronList += c.neuronList
        if hasattr(self, 'connList'):
            for c in self.child:
                if (c is not None):
                    self.connList += c.connList


class Root():

    def __init__(self, configs=None):
        self.ID = ''
        self.depth = 0
        if (configs is None):
            self.configs = {
                'input_size': 4,
                'output_size': 2
            }
        else:
            self.configs = configs

    def generate(self):
        self.child = [Div(self), Conns(self)]
        self.child[0].generate()
        self.child[1].generate()


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
        if (self.rule == 3):
            self.child[1].compile()
            self.child[0].compile()
            self.child[2].compile()
        else:
            self.child[0].compile()
            self.child[1].compile()
            self.child[2].compile()
        self.merge_from_children()


class Clones(TreeNode):

    def __init__(self, parent, sibling):
        super().__init__(parent)
        self.sibling = sibling

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
        if (self.rule == 0):
            self.child[0].compile()
            self.child[1].compile()
        else:
            self.child[0].compile()
        merge_from_children()


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

    def compile(self):
