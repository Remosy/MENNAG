import networkx as nx
import matplotlib.pyplot as plt
import pdb
import nodes


def node_display_name(node):
    return node.__class__.__name__ + '\n ' + node.ID


def visit_tree(pointer, vertices, edges):
    vertices.append(node_display_name(pointer))
    for i in range(2):
        if (not isinstance(pointer, nodes.Cell)):
            if (pointer.child[i] is not None):
                edges.append((
                    node_display_name(pointer),
                    node_display_name(pointer.child[i]))
                )
                visit_tree(pointer.child[i], vertices, edges)


class IndividualVisualizer():

    def __init__(self, individual):
        self.individual = individual

    def draw_tree(self):
        pointer = self.individual.child[0]
        vertices = []
        edges = []
        visit_tree(pointer, vertices, edges)
        # pdb.set_trace()
        treeGraph = nx.DiGraph()
        treeGraph.add_nodes_from(vertices)
        treeGraph.add_edges_from(edges)
        layout = nx.nx_agraph.graphviz_layout(
            treeGraph,
            prog='dot',
            args='-Gnodesep=10')
        plt.figure(1, figsize=(20, 20))
        nx.draw(treeGraph, layout, with_labels=True, node_size=1000)

    def draw_nn(self):
        vertices = self.individual.nn.nodeNames
        edges = self.individual.nn.rawConnList
        nnGraph = nx.DiGraph()
        nnGraph.add_nodes_from(vertices)
        nnGraph.add_weighted_edges_from(edges)
        layout = nx.nx_agraph.graphviz_layout(
            nnGraph,
            prog='dot',
            args='-Gnodesep=10')
        plt.figure(1, figsize=(20, 20))
        nx.draw(nnGraph, layout, with_labels=True, node_size=1000)
        labels = nx.get_edge_attributes(nnGraph, 'weight')
        nx.draw_networkx_edge_labels(nnGraph, layout, edge_labels=labels)
        plt.savefig('test.png')
