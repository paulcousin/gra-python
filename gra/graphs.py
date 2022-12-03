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


################################ CLASS ################################

class Graph:

    # initialize object after checking input validity
    def __init__(self, adjacency_matrix, state_vector, force=False):

        # force the adjacency_matrix and state_vector attributes (useful for imports)
        if force:  
            self.dtype = None
            self.adjacency_matrix = adjacency_matrix
            self.state_vector = state_vector
            return

        # raise error if improperly formated input
        # then chose the right data type
        # then create dtype attribute
        ## check types
        if not (type(adjacency_matrix) == type(state_vector) == list):
            raise TypeError('Adjacency matrix and state vector must be provided as nested lists.')
        ## check dimensions
        if not (len(state_vector) == len(adjacency_matrix) and all([len(i) == len(state_vector) for i in adjacency_matrix])):
            raise TypeError('Adjacency matrix and state vector must have compatible dimensions.')
        ## check content
        if all([type(i[0]) == int for i in state_vector]) and all([all([type(i) == int for i in j]) for j in adjacency_matrix]):
            self.dtype = np.int32
        elif all([type(i[0]) == int or type(i[0]) == float for i in state_vector]) and all([all([type(i) == int or type(i) == float for i in j]) for j in adjacency_matrix]):
            self.dtype = np.float32
        else:
            raise TypeError('Adjacency matrix and state vector must contain real values only.')

        # create adjacency_matrix and state_vector attributes
        self.adjacency_matrix = tf.sparse.from_dense(tf.constant(adjacency_matrix, dtype=self.dtype))
        self.state_vector = tf.constant(state_vector, dtype=self.dtype)
    
    #--------------- UTILITIES ---------------#
    
    # A==B : check if A and B are identical whatever the order of their vertices
    def __eq__(self, g2):
        ig1 = self.to_igraph()  # A
        ig2 = g2.to_igraph()    # B
        isomorphisms = ig1.get_isomorphisms_vf2(ig2) # use igraph to fetch isomorphisms
        test = np.zeros(self.order()) # initiate test vector

        for i in range(len(isomorphisms)): # check every combination
            for j in range(len(isomorphisms[i])):
                test[j] = ig1.vs["label"][isomorphisms[i][j]]
            if (test == ig2.vs["label"]).all(): 
                return True # if an valid equavallence was found
        
        return False # otherwise
    
    def order(self): # return the order of the graph
        return self.adjacency_matrix.dense_shape.numpy()[1]
    
    def diameter(self): # return the diameter of the graph
        return nx.diameter(self.to_networkx())
    
    def clone(self): # create a clone
        return copy.deepcopy(self)

    #--------------- EVOLUTION METHOD ---------------#

    # evolve graph with rule 1 time
    def evolve(self, rule): 
        rule(self)
        return self
    
    # evolve graph with rule n times
    def jump(self, rule, n):
        for i in range(n):
            rule(self)
        return self
    
    #--------------- GRAPH PLOT ---------------#

    # plot the graph using igraph
    # the area of the plot groes linearly with the number of vertices
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

    # to igraph
    def to_igraph(self):
        edgelist = self.adjacency_matrix.indices.numpy()
        g = ig.Graph(n=self.order(), edges=edgelist)
        g.vs["label"] = [self.state_vector.numpy()[d][0] for d in range(self.order())]
        return g.simplify()
    
    # to networkx
    def to_networkx(self):
        g = nx.Graph()
        g.add_edges_from(self.adjacency_matrix.indices.numpy())
        for i in range(self.order()):
            g.add_node(i, value=self.state_vector.numpy()[i][0])
        return g

    # to Mathematica
    ### return a string compatible with the Mathematica GRA package:
    ### https://github.com/paulcousin/gra-mathematica
    def to_mathematica(self):
        aM = "SparseArray[{"+','.join([str(list(d))+"->1" for d in self.adjacency_matrix.indices.numpy()+1]).replace('[','{').replace(']','}')+"},{"+','.join([str(d) for d in self.adjacency_matrix.dense_shape.numpy()])+"}]"
        sV = "{"+','.join([str(d) for d in self.state_vector.numpy()]).replace('[','{').replace(']','}')+"}"
        return "{"+aM+","+sV+"}"


################################ FUNCTIONS ################################

# import in gra a graph in the igraph format
def from_igraph(igraph):

    # search for the right format to apply and create the graph object
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

    # create the list of indices
    indices = [list(i) for i in igraph.get_edgelist()]

    # create a sparse tensor directly
    adjacency_matrix = tf.sparse.SparseTensor(
        indices = indices, 
        values = tf.ones(len(indices), dtype=graph.dtype), 
        dense_shape = [igraph.vcount(),igraph.vcount()]
    )

    # update the graph object
    graph.adjacency_matrix = tf.sparse.reorder(tf.sparse.add(adjacency_matrix, tf.sparse.transpose(adjacency_matrix)))
    graph.state_vector = tf.constant(state_vector, dtype=graph.dtype)

    return graph

# This function goes through all possible minimal d-regular graphs and returns the complete list.
# It imposes that:
#  * the vertex number n corresponds to the order-n vertex in the color-symmetric graph
#  * the order of the graph is 2*(degree+1)+2  (required for minimality)
#  * the first vertex is alive and all its neighbors are alive
#  * the second vertex is alive and all of its neighbors are dead
#  And searches the space of possible graphs to find the ones with the desired properties.
def minimal_regular_graphs(degree):
    order = 2*(degree+1)+2
    half = degree+2
    size = degree**2
    state_vector = np.array([[1] if i<order/2 else [0] for i in range(order)])
    ones = np.ones([order,1], dtype=int)
    results=[]

    # initiates progress
    update = 0
    step = max([int(2**size/100),1])
    print("0% -> 0 graphs found", end="\r")

    for i in range(2**size):
        # display progress
        if i//step != update : 
            update = i//step
            print (str(int((i/(2**size))*100)) + "% -> " + str(len(results)) + " graphs found", end="\r")

        A = [int(x) for x in np.binary_repr(i)]
        A.reverse()
        for j in range(len(A), size): A.append(0)
        A = np.array(A)
        A.shape = (half-2, half-2)

        initial_matrix = np.zeros((order, order), dtype=int)
        initial_matrix[0,2:half]=np.ones(half-2)
        initial_matrix[1,half:order-1]=np.ones(half-1)
        initial_matrix[2:half,2:half]=np.triu(A.T, k=1)
        initial_matrix[2:half,half:order-2]=np.flip(np.triu(A, k=0).T, 0)
        initial_matrix = initial_matrix + initial_matrix.T
        initial_matrix =  initial_matrix + np.rot90(np.triu(np.rot90(initial_matrix,-1), k=1),-1)   

        for p in range(half-1):
            matrix = copy.deepcopy(initial_matrix)
            matrix[1,half+p]=0
            matrix[half+p,1]=0
            matrix[-2,-half-p-1]=0
            matrix[-half-p-1,-2]=0

            if all(np.dot(matrix, ones)==degree*ones):
                C = (degree+1)*state_vector + np.dot(matrix,state_vector)
                test = True
                for k in range(2*(degree+1)):
                    if k not in C:
                        test = False

                if test:
                    graph = gra.Graph(matrix.tolist(),state_vector.tolist())
                    if graph not in results:
                        results.append(graph)

    print("100% -> " + str(len(results)) + " graphs found", end="\r")

    return results
