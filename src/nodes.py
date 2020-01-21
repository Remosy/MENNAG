import random
import numpy as np
import configs
import activations
from nn.feedforward import FeedForward
import pdb


def get_all_nodes_with_rule(pointer, type, rule, nodeList):
    if ((isinstance(pointer, type)) and (pointer.rule == rule)):
        nodeList.append(pointer)
    for c in pointer.child:
        if (c is not None):
            get_all_nodes_with_rule(c, type, rule, nodeList)


def get_all_nodes(pointer, type, nodeList):
    if (isinstance(pointer, type)):
        nodeList.append(pointer)
    for c in pointer.child:
        if (c is not None):
            get_all_nodes(c, type, nodeList)


class Root():

    def __init__(self, dict=None):
        self.ID = ''
        self.depth = 0
        if (dict is None):
            dict = {
                'max_depth': 5,
                'input_size': 4,
                'output_size': 2,
                'feedforward': True,
                'forward_prob': 0.8,
                'weight_mean': 0,
                'weight_std': 1,
                'perturb_std': 0.2,
                'cross_rate': 0.3,
                'weight_perturb_rate': 0.3,
                'weight_reset_rate': 0.05,
                'insertion_rate': 0.2,
                'deletion_rate': 0.1,
                'random_tree_rate': 0.05
            }
        self.configs = configs.Configs(dict)
        if (self.configs.feedforward):
            self.nn = FeedForward(self.configs)

    def generate(self):
        self.child = [Div(self), None, None]
        self.child[0].generate()

    def compile(self):
        self.nn = FeedForward(self.configs)
        self.child[0].ID = ''
        self.child[0].compile()
        self.neuronSet = self.child[0].neuronSet
        self.connSet = self.child[0].connSet
        self.inSet = self.child[0].inSet
        self.outSet = self.child[0].outSet
        self.nodeSet = self.child[0].nodeSet
        for neuron in self.neuronSet:
            self.nn.add_node(neuron, activations.sigmoid)
        for conn in self.connSet:
            source = conn[0]
            target = conn[1]
            while (source not in self.neuronSet and source != ''):
                source = source[:len(source) - 1]
            while (target not in self.neuronSet and target != ''):
                target = target[:len(target) - 1]
            if (not(source == '' or target == '')):
                self.nn.add_conn(source, target, conn[2])
        for inConn in self.inSet:
            self.nn.add_conn('i' + str(inConn[0]), inConn[1], inConn[2])
        for outConn in self.outSet:
            self.nn.add_conn(outConn[0], 'o' + str(outConn[1]), outConn[2])
        return self.nn

    def get_div(self, ID):
        pointer = self.child[0]
        for i in range(len(ID)):
            pointer = pointer.child[int(ID[i])]
            if (pointer is None):
                return None
        return pointer

    def get_all_nodes(self, type=None, rule=None):
        candidates = []
        for n in self.nodeSet:
            if ((type is None) or (isinstance(n, type))):
                if ((rule is None) or (n.rule == rule)):
                    candidates.append(n)
        return candidates

    def insert_at_div(self, type):
        candidates = []
        while(len(candidates) < 1):
            selectRule = random.choice([0, 1, 2])
            candidates = self.get_all_nodes_with_rule(type=Div, rule=selectRule)
        insertAt = random.choice(candidates)
        if (selectRule == 1):
            insertPos = 0
        if (selectRule == 2):
            insertPos = 1
        elif (selectRule == 0):
            insertPos = random.choice([0, 1])
        n1 = insertAt.child[insertPos]
        n2 = Div(insertAt)
        if (type == Div):
            n2.rule = 0
            newPos = random.choice([0, 1])
            oldPos = 1 - newPos
        else:
            n2.rule = random.choice([1, 2])
            if (n2.rule == 1):
                newPos = 1
                oldPos = 0
            else:
                newPos = 0
                oldPos = 1
        n2.child[oldPos] = n1
        n1.parent = n2
        n2.child[newPos] = type(n2)
        n2.child[newPos].generate()
        n2.child[2] = Conns(n2)
        insertAt.child[insertPos] = n2

    def delete_at_div(self, type):
        candidates = []
        if (type == Div):
            selectRule = 0
        else:
            selectRule = random.choice([1, 2])
        candidates = self.get_all_nodes(type=Div, rule=selectRule)
        if (len(candidates) == 0):
            return
        selected = random.choice(candidates)
        selectedPos = 0
        for i in range(3):
            if (selected.parent.child[i] == selected):
                selectedPos = i
        if (selectRule == 0):
            deletePos = random.choice([0, 1])
        elif (selectRule == 1):
            deletePos = 1
        elif (selectRule == 2):
            deletePos = 0
        siblingPos = 1 - deletePos
        deleted = selected.child[deletePos]
        sibling = selected.child[siblingPos]
        sibling.parent = selected.parent
        sibling.parent.child[selectedPos] = sibling

    def insert_at_list(self, listType):
        if (listType == Clones):
            type = Clone
        elif (listType == Conns):
            type = Conn
        candidates = self.get_all_nodes(type=listType)
        insertAt = random.choice(candidates)
        listNode = listType(insertAt.parent)
        if (insertAt.parent.child[0] == insertAt):
            insertAt.parent.child[0] = listNode
        else:
            insertAt.parent.child[1] = listNode
        insertAt.parent = listNode
        listNode.child[1] = insertAt
        listNode.child[0] = type(listNode)
        listNode.child[0].generate()
        listNode.rule = 0

    def cross_at(self, Div1, Div2):
        Div2 = Div2.deepcopy(Div1.parent)
        if (Div1.parent.child[0] == Div1):
            Div1.parent.child[0] = Div2
        else:
            Div1.parent.child[1] = Div2


class TreeNode():

    def __init__(self, parent):
        self.parent = parent
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
        self.nodeSet = set([self])

    def mutate(self):
        return

    def generate(self):
        for c in self.child:
            if (c is not None):
                c.generate()
        return

    def compile(self):
        self.reset()
        for c in self.child:
            if (c is not None):
                c.compile()
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
            leftDepth = self.child[0].max_depth_from_current()
        else:
            leftDepth = self.depth
        if (self.child[1] is not None):
            rightDepth = self.child[1].max_depth_from_current()
        else:
            rightDepth = self.depth
        return max(leftDepth, rightDepth)

    def merge_from_children(self):
        for c in self.child:
            if (c is not None):
                self.neuronSet.update(c.neuronSet)
                self.connSet.update(c.connSet)
                self.outSet.update(c.outSet)
                self.inSet.update(c.inSet)
                self.nodeSet.update(c.nodeSet)

    def deepcopy(self, newParent):
        copy = self.__class__(newParent)
        copy.rule = self.rule
        for i in range(3):
            if (self.child[i] is not None):
                copy.child[i] = self.child[i].deepcopy(copy)
        return copy


class Div(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)
        self.depth += 1

    def generate(self):
        if (self.depth >= self.configs.max_depth):
            # Rule #3 DIV -> CELL CELL
            self.rule = 3
        else:
            number = random.random()
            if (number < 0.2):
                if (random.random() < 0.5):
                    # Rule #1 DIV -> DIV CLONES CONNS
                    self.rule = 1
                else:
                    # Rule #2 DIV -> CLONES DIV CONNS
                    self.rule = 2
            elif (number < 1 / (self.depth**0.5)):
                # Rule #0 DIV -> DIV DIV CONNS
                self.rule = 0
            else:
                # Rule #3 DIV -> CELL CELL
                self.rule = 3
        self.generate_by_rule()
        super().generate()

    def generate_by_rule(self):
        if (self.rule == 0):
            self.child[0] = Div(self)
            self.child[1] = Div(self)
            self.child[2] = Conns(self)
        elif (self.rule == 1):
            self.child[0] = Div(self)
            self.child[1] = Clones(self)
            self.child[2] = Conns(self)
        elif (self.rule == 2):
            self.child[1] = Div(self)
            self.child[0] = Clones(self)
            self.child[2] = Conns(self)
        elif (self.rule == 3):
            self.generate_cells()

    def generate_cells(self):
        # Rule #3 DIV -> CELL CELL CONNS
        self.child[0] = Cell(self)
        self.child[1] = Cell(self)
        self.child[2] = Conns(self)

    def compile(self):
        self.reset()
        self.child[0].ID = self.ID + '0'
        self.child[1].ID = self.ID + '1'
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
        for c in self.child:
            if (c is not None):
                c.update_depth()
        return


class Clones(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)
        self.depth += 1

    def generate(self):
        number = random.random()
        if (number < 0.01):
            # rule #0 CLONES -> CLONES CLONE
            self.child[0] = Clone(self)
            self.child[1] = Clones(self)
            self.rule = 0
        else:
            # rule #1 CLONES -> CLONE
            self.child[0] = Clone(self)
            self.rule = 1
        super().generate()

    def compile(self):
        self.reset()
        if (isinstance(self.parent, Div)):
            if (self.parent.rule == 1):
                self.sibling = self.parent.child[0]
            else:
                self.sibling = self.parent.child[1]
        else:
            self.sibling = self.parent
        if (self.rule == 0):
            self.child[0].ID = self.ID + '0'
            self.child[1].ID = self.ID + '1'
            self.child[0].compile()
            self.child[1].compile()
        else:
            self.child[0].ID = self.ID
            self.child[0].compile()
        self.merge_from_children()

    def update_depth(self):
        self.depth = self.parent.depth + 1
        for c in self.child:
            if (c is not None):
                c.update_depth()
        return


class Clone(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        self.permi = np.random.rand(self.configs.input_size)
        self.permo = np.random.rand(self.configs.output_size)

    def compile(self):
        self.reset()
        self.sibling = self.parent.sibling
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

    def deepcopy(self, newParent):
        copy = Clone(newParent)
        copy.permi = self.permi.copy()
        copy.permo = self.permo.copy()
        return copy


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

    def compile(self):
        self.ID = self.parent.ID
        super().compile()


class Conn(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        max_depth = self.configs.max_depth
        self.sourceTail = bin(random.getrandbits(max_depth))[2:]
        self.targetTail = bin(random.getrandbits(max_depth))[2:]
        if (self.configs.feedforward):
            # feedforward connection
            self.sourceAddon = '0'
            self.targetAddon = '1'
        else:
            number = random.random()
            if (number < self.configs.forward_prob):
                self.sourceAddon = '0'
                self.targetAddon = '1'
            else:
                self.sourceAddon = ''
                self.targetAddon = ''

        mean = self.configs.weight_mean
        std = self.configs.weight_std
        self.weight = np.random.normal(mean, std)

    def compile(self):
        self.reset()
        self.ID = self.parent.ID
        d = self.max_depth_from_current()
        source = self.ID + self.sourceAddon + self.sourceTail
        source = source[:d]
        target = self.ID + self.targetAddon + self.targetTail
        target = target[:d]
        self.connSet.add((source, target, self.weight))

    def max_depth_from_current(self):
        p = self.parent
        while(isinstance(p, Conns)):
            p = p.parent
        return p.max_depth_from_current()

    def deepcopy(self, newParent):
        copy = Conn(newParent)
        copy.sourceAddon = self.sourceAddon
        copy.targetAddon = self.sourceAddon
        copy.sourceTail = self.sourceTail
        copy.targetTail = self.targetTail
        copy.weight = self.weight
        return copy


class Cell(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        self.child[0] = In(self)
        self.child[1] = Out(self)
        super().generate()

    def compile(self):
        self.reset()
        self.neuronSet.add(self.ID)
        self.child[0].ID = self.ID
        self.child[1].ID = self.ID
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
            self.inSet.add((self.source, self.ID, self.weight))

    def deepcopy(self, newParent):
        copy = In(newParent)
        copy.rule = self.rule
        if (self.rule == 1):
            copy.source = self.source
            copy.weight = self.weight
        return copy


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

    def compile(self):
        self.reset()
        if (self.rule == 1):
            self.outSet.add((self.ID, self.target, self.weight))

    def deepcopy(self, newParent):
        copy = Out(newParent)
        copy.rule = self.rule
        if (self.rule == 1):
            copy.target = self.target
            copy.weight = self.weight
        return copy
