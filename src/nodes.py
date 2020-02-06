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

    def __init__(self, config=None):
        self.ID = ''
        self.depth = -1
        self.config = config
        if (self.config.feedforward):
            self.nn = FeedForward(self.config)

    def generate(self):
        self.child = [Div(self), None, None]
        self.child[0].generate()

    def compile(self):
        self.child[0].ID = ''
        self.child[0].update_depth()
        self.child[0].compile()
        self.neuronSet = self.child[0].neuronSet
        self.connSet = self.child[0].connSet
        self.inSet = self.child[0].inSet
        self.outSet = self.child[0].outSet
        self.nodeSet = self.child[0].nodeSet

    def execute(self):
        self.nn = FeedForward(self.config)
        for neuron in self.neuronSet:
            self.nn.add_node(neuron, 1)
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
        self.nn.compile()
        return self.nn

    def mutate(self):
        n0 = random.random()
        insertionThreshold = self.config.insertion_rate
        deletionThreshold = insertionThreshold + self.config.deletion_rate
        if (n0 < insertionThreshold):
            n1 = random.random()
            if (n1 < 0.5):
                type = random.choice([Div, Clones])
                self.insert_at_div(type)
            else:
                type = random.choice([Clone, Conn])
                self.insert_at_list(type)
        elif (n0 < deletionThreshold):
            n1 = random.random()
            if (n1 < 0.5):
                type = random.choice([Div, Clones])
                self.delete_at_div(type)
            else:
                type = random.choice([Clone, Conn])
                self.delete_at_list(type)
        else:
            self.child[0].mutate()

    def get_node(self, ID):
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
        availRules = [0, 1, 2]
        while((len(candidates) < 1) and (len(availRules) > 1)):
            selectRule = random.choice(availRules)
            candidates = self.get_all_nodes(type=Div, rule=selectRule)
            availRules.remove(selectRule)
        if (len(candidates) == 0):
            return 0
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
        n2.child[2].generate()
        insertAt.child[insertPos] = n2

    def delete_at_div(self, type):
        candidates = []
        if (type == Div):
            selectRule = 0
        elif (type == Clones):
            selectRule = random.choice([1, 2])
        candidates = self.get_all_nodes(type=Div, rule=selectRule)
        if (len(candidates) == 0):
            return 0
        selected = random.choice(candidates)
        if (selected == self.child[0]):
            return
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
        return 1

    def insert_at_list(self, type):
        if (type == Clone):
            listType = Clones
        elif (type == Conn):
            listType = Conns
        candidates = self.get_all_nodes(type=listType)
        if (len(candidates) == 0):
            return
        insertAt = random.choice(candidates)
        listNode = listType(insertAt.parent)
        insertAt.parent.child[insertAt.get_pos()] = listNode
        insertAt.parent = listNode
        listNode.child[1] = insertAt
        listNode.child[0] = type(listNode)
        listNode.child[0].generate()
        listNode.rule = 0

    def delete_at_list(self, type):
        if (type == Clone):
            listType = Clones
        elif (type == Conn):
            listType = Conns
        candidates = self.get_all_nodes(type=listType, rule=0)
        if (len(candidates) == 0):
            return 0
        deleted = random.choice(candidates)
        deleted.parent.child[deleted.get_pos()] = deleted.child[1]
        deleted.child[1].parent = deleted.parent
        return 1

    def cross_with(self, p2):
        offspring = self.deepcopy()
        p1nodes = offspring.get_all_nodes(type=Div)
        p2nodes = p2.get_all_nodes(type=Div)
        Div1 = random.choice(p1nodes)
        Div2 = random.choice(p2nodes).deepcopy(Div1.parent)
        if (Div1.parent.child[0] == Div1):
            Div1.parent.child[0] = Div2
        else:
            Div1.parent.child[1] = Div2
        return offspring

    def deepcopy(self):
        copy = Root(config=self.config)
        copy.child = [self.child[0].deepcopy(copy), None, None]
        copy.compile()
        return copy


class TreeNode():

    def __init__(self, parent):
        self.parent = parent
        self.depth = parent.depth
        self.config = parent.config
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
        for c in self.child:
            if (c is not None):
                c.mutate()
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

    def get_pos(self):
        for i in range(3):
            if (self.parent.child[i] == self):
                return i

    def deepcopy(self, newParent):
        copy = self.__class__(newParent)
        copy.rule = self.rule
        copy.depth = self.depth
        copy.ID = self.ID
        for i in range(3):
            if (self.child[i] is not None):
                copy.child[i] = self.child[i].deepcopy(copy)
        return copy


class Div(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)
        self.depth += 1

    def generate(self):
        if (self.depth >= self.config.max_depth):
            # Rule #3 DIV -> CELL CELL
            self.rule = 3
        else:
            if (random.random() < 0.01):
                if (random.random() < 0.5):
                    # Rule #1 DIV -> DIV CLONES CONNS
                    self.rule = 1
                else:
                    # Rule #2 DIV -> CLONES DIV CONNS
                    self.rule = 2
            elif (random.random() < 1 / (self.depth**2 + 1e-5)):
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
        if (self.rule == 2):
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
            self.sibling = self.parent.sibling
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
        self.permi = np.random.rand(self.config.input_size)
        self.permo = np.random.rand(self.config.output_size)

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

    def mutate(self):
        pos = random.choice(range(len(self.permi)))
        self.permi[pos] = np.random.rand()
        pos = random.choice(range(len(self.permo)))
        self.permo[pos] = np.random.rand()

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
            # Rule 0 CONNS -> CONN CONNS
            self.child[0] = Conn(self)
            self.child[1] = Conns(self)
            self.rule = 0
        else:
            # Rule 1 CONNS -> CONN
            self.rule = 1
        super().generate()

    def compile(self):
        self.ID = self.parent.ID
        super().compile()


class Conn(TreeNode):

    def __init__(self, parent):
        super().__init__(parent)

    def generate(self):
        max_depth = self.config.max_depth
        self.sourceTail = bin(random.getrandbits(max_depth))[2:]
        self.targetTail = bin(random.getrandbits(max_depth))[2:]
        if (self.config.feedforward):
            # feedforward connection
            self.sourceAddon = '0'
            self.targetAddon = '1'
        else:
            number = random.random()
            if (number < self.config.forward_prob):
                self.sourceAddon = '0'
                self.targetAddon = '1'
            else:
                self.sourceAddon = ''
                self.targetAddon = ''

        mean = self.config.weight_mean
        std = self.config.weight_std
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

    def mutate(self):
        number = random.random()
        resetThreshold = self.config.weight_reset_rate
        perturbThreshold = resetThreshold + self.config.weight_perturb_rate
        if (number < resetThreshold):
            self.generate()
        elif (number < perturbThreshold):
            self.weight += np.random.normal(0, self.config.perturb_std)

    def deepcopy(self, newParent):
        copy = Conn(newParent)
        copy.sourceAddon = self.sourceAddon
        copy.targetAddon = self.targetAddon
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
            self.source = random.choice(range(0, self.config.input_size))
            mean = self.config.weight_mean
            std = self.config.weight_std
            self.weight = np.random.normal(mean, std)

    def compile(self):
        self.reset()
        if (self.rule == 1):
            self.inSet.add((self.source, self.ID, self.weight))

    def mutate(self):
        number = random.random()
        resetThreshold = self.config.weight_reset_rate
        perturbThreshold = resetThreshold + self.config.weight_perturb_rate
        if (number < resetThreshold):
            self.generate()
        elif (number < perturbThreshold):
            if (self.rule == 1):
                self.weight += np.random.normal(0, self.config.perturb_std)

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
            self.target = random.choice(range(0, self.config.output_size))
            mean = self.config.weight_mean
            std = self.config.weight_std
            self.weight = np.random.normal(mean, std)

    def compile(self):
        self.reset()
        if (self.rule == 1):
            self.outSet.add((self.ID, self.target, self.weight))

    def mutate(self):
        number = random.random()
        resetThreshold = self.config.weight_reset_rate
        perturbThreshold = resetThreshold + self.config.weight_perturb_rate
        if (number < resetThreshold):
            self.generate()
        elif (number < perturbThreshold):
            if (self.rule == 1):
                self.weight += np.random.normal(0, self.config.perturb_std)

    def deepcopy(self, newParent):
        copy = Out(newParent)
        copy.rule = self.rule
        if (self.rule == 1):
            copy.target = self.target
            copy.weight = self.weight
        return copy
