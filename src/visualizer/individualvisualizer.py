import networkx as nx
import matplotlib.pyplot as plt

class IndividualVisualizer():

    def __init__(self, individual):
        self.individual = individual

    def draw_nn(self):
        vertices = self.individual.nn.nodeNames
        edges = self.individual.nn.rawConnList
        nnGraph = nx.DiGraph()
        nnGraph.add_nodes_from(vertices)
        nnGraph.add_weighted_edges_from(edges)
        nx.draw_spring(nnGraph, with_labels=True, font_weight='bold')
        self.nnGraph = nnGraph
