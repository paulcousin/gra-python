# packages
import gra
import math
import copy
import numpy as np
import igraph as ig
import networkx as nx
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # removes unnecessary outputs from TensorFlow
import tensorflow as tf


################################ CLASSES ################################

class Graph:

    def __init__(self, adjacency_matrix, state_vector, force=False):
        if force:  # force the adjacency_matrix and state_vector attributes
            self.dtype = None
            self.adjacency_matrix = adjacency_matrix
            self.state_vector = state_vector
            return

        # check types
        if not (type(adjacency_matrix) == type(state_vector) == list):
            raise TypeError('Adjacency matrix and state vector must be provided as nested lists.')
        # check dimensions
        if not (len(state_vector) == len(adjacency_matrix) and all([len(i) == len(state_vector) for i in adjacency_matrix])):
            raise TypeError('Adjacency matrix and state vector must have compatible dimensions.')
        # check content
        if all([type(i[0]) == int for i in state_vector]) and all([all([type(i) == int for i in j]) for j in adjacency_matrix]):
            self.dtype = np.int32
        elif all([type(i[0]) == int or type(i[0]) == float for i in state_vector]) and all([all([type(i) == int or type(i) == float for i in j]) for j in adjacency_matrix]):
            self.dtype = np.float32
        else:
            raise TypeError('Adjacency matrix and state vector must contain real values only.')

        # create attributes
        self.adjacency_matrix = tf.sparse.from_dense(tf.constant(adjacency_matrix, dtype=self.dtype))
        self.state_vector = tf.constant(state_vector, dtype=self.dtype)
    
    #--------------- UTILITIES ---------------#
    def __eq__(self, g2):
        g1 = self
        ig1 = self.to_igraph()
        ig2 = g2.to_igraph()

        isomorphisms = ig1.get_isomorphisms_vf2(ig2)
        state_vector = g1.state_vector.numpy()
        test = g2.state_vector.numpy()

        for i in range(len(isomorphisms)):
            for j in range(len(isomorphisms[i])):
                test[j] = g2.state_vector.numpy()[isomorphisms[i][j]]
            if (test == state_vector).all(): 
                return True
        
        return False
    
    def order(self):
        return self.adjacency_matrix.dense_shape.numpy()[1]
    
    def diameter(self):
        return nx.diameter(self.to_networkx())
    
    def clone(self):
        return copy.deepcopy(self)

    #--------------- EVOLUTION METHOD ---------------#
    def evolve(self, rule): 
        rule(self)
        return self
    
    def jump(self, rule, n):
        for i in range(n):
            rule(self)
        return self
    
    #--------------- GRAPH PLOT ---------------#
    def plot(self):
        edgelist = self.adjacency_matrix.indices.numpy()
        g = ig.Graph(n=self.order(), edges=edgelist).simplify()
        visual_style = {
            "vertex_size": 4,
            "layout": g.layout_kamada_kawai(maxiter=10*self.order())
            }
        if all([i == [0] or i == [1] for i in self.state_vector.numpy()]):
            visual_style["vertex_color"] = ["purple" if self.state_vector.numpy()[d][0]==1 else "orange" for d in range(self.order())]
        return ig.plot(g, bbox=(20*math.sqrt(self.order()), 20*math.sqrt(self.order())), margin=10, **visual_style)

    #--------------- EXPORTS ---------------#
    def to_igraph(self):
        edgelist = self.adjacency_matrix.indices.numpy()
        g = ig.Graph(n=self.order(), edges=edgelist)
        g.vs["label"] = [self.state_vector.numpy()[d][0] for d in range(self.order())]
        return g.simplify()
    
    def to_networkx(self):
        g = nx.Graph()
        g.add_edges_from(self.adjacency_matrix.indices.numpy())
        for i in range(self.order()):
            g.add_node(i, value=self.state_vector.numpy()[i][0])
        return g

    def to_mathematica(self):
        aM = "SparseArray[{"+','.join([str(list(d))+"->1" for d in self.adjacency_matrix.indices.numpy()+1]).replace('[','{').replace(']','}')+"},{"+','.join([str(d) for d in self.adjacency_matrix.dense_shape.numpy()])+"}]"
        sV = "{"+','.join([str(d) for d in self.state_vector.numpy()]).replace('[','{').replace(']','}')+"}"
        return "{"+aM+","+sV+"}"


################################ FUNCTIONS ################################
def from_igraph(igraph):

    if all([type(i) == np.int32 for i in igraph.vs['label']]):
        state_vector = [[int(i)] for i in igraph.vs['label']]
        graph = gra.Graph(None,None,force=True)
        graph.dtype = np.int32
    elif all([type(i) == np.float32 or type(i) == np.int32 for i in igraph.vs['label']]):
        state_vector = [[float(i)] for i in igraph.vs['label']]
        graph = gra.Graph(None,None,force=True)
        graph.dtype = np.float32
    else:
        raise TypeError('Improperly formated graph.')

    indices = [list(i) for i in igraph.get_edgelist()]

    adjacency_matrix = tf.sparse.SparseTensor(
        indices = indices, 
        values = tf.ones(len(indices), dtype=graph.dtype), 
        dense_shape = [igraph.vcount(),igraph.vcount()]
    )

    graph.adjacency_matrix = tf.sparse.reorder(tf.sparse.add(adjacency_matrix, tf.sparse.transpose(adjacency_matrix)))
    graph.state_vector = tf.constant(state_vector, dtype=graph.dtype)

    return graph
