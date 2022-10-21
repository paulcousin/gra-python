# packages
import gra
import math
import numpy as np
import igraph as ig
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # removes unnecessary outputs from TensorFlow
import tensorflow as tf

class Graph:
   
    #--------------- INITIALIZATION METHOD ---------------#
    def __init__(self, adjacency_matrix, state_vector):
        # checks the format of the inputs and acts accordingly to create two properly formated attributes
        if tf.is_tensor(adjacency_matrix): # if adjacency_matrix is a tensor
            self.adjacency_matrix = adjacency_matrix
        else: # if adjacency_matrix is not a tensor
            self.adjacency_matrix = tf.sparse.from_dense(tf.constant(adjacency_matrix, dtype=tf.int32))
        if tf.is_tensor(state_vector): # if state_vector is a tensor
            self.state_vector = state_vector
        else: # if state_vector is not a tensor
            self.state_vector = tf.constant(state_vector, dtype=tf.int32)
            

    #--------------- UTILITIES ---------------#
    def __eq__(self, g2):
        g1 = self
        ig1 = self.igraph()
        ig2 = g2.igraph()

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
    
    def clone(self):
        return gra.Graph(self.adjacency_matrix, self.state_vector)

    #--------------- GRAPH PLOT ---------------#
    def plot(self):
        edgelist = np.asarray([list(d) for d in self.adjacency_matrix.indices.numpy()], dtype= np.int32) # TO SIMPLIFY LATER
        g = ig.Graph(n=self.order(), edges=edgelist).simplify()
        visual_style = {}
        visual_style["vertex_size"] = 4
        visual_style["layout"] = g.layout_kamada_kawai(maxiter=10*self.order())
        visual_style["vertex_color"] = ["purple" if self.state_vector.numpy()[d][0]==1 else "orange" for d in range(self.order())]
        return ig.plot(g, bbox=(20*math.sqrt(self.order()), 20*math.sqrt(self.order())), margin=10, **visual_style)

    #--------------- EVOLUTION METHOD ---------------#
    def evolve(self, rule): 
        rule(self)
    
    def jump(self, rule, n):
        for i in range(n):
            rule(self)

    #--------------- EXPORTS ---------------#
    def mathematica(self):
        aM = "SparseArray[{"+','.join([str(list(d))+"->1" for d in self.adjacency_matrix.indices.numpy()+1]).replace('[','{').replace(']','}')+"},{"+','.join([str(d) for d in self.adjacency_matrix.dense_shape.numpy()])+"}]"
        sV = "{"+','.join([str(d) for d in self.state_vector.numpy()]).replace('[','{').replace(']','}')+"}"
        return "{"+aM+","+sV+"}"
        
    def igraph(self):
        edgelist = np.asarray([list(d) for d in self.adjacency_matrix.indices.numpy()], dtype= np.int32) # TO SIMPLIFY LATER
        g = ig.Graph(n=self.order(), edges=edgelist).simplify()
        g.vs["size"] = 4
        g.vs["color"] = ["purple" if self.state_vector.numpy()[d][0]==1 else "orange" for d in range(self.order())]
        return g
