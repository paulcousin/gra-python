# external packages
import gra
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # removes unnecessary outputs from TensorFlow
import tensorflow as tf

class Rule:

    #--------------- INITIALIZATION METHOD ---------------#
    def __init__(self, degree, number):
        if 0 <= int(number) < 4**(2*(int(degree)+1)) and int(degree) > 0:
            self.degree = int(degree)
            self.number = int(number)
        else:
            raise TypeError('Invalid input.')
    
    #--------------- UTILITIES ---------------#
    def binary_digits(self):
        # creates a list with the 4*(d+1) first binary digits of the rule number
        rule = [int(x) for x in np.binary_repr(self.number)]
        rule.reverse()
        for i in range(len(rule), 4*(self.degree+1)): rule.append(0)
        return rule[::-1]

    #--------------- EVOLUTION METHODS ---------------#
    def __call__(self, graph):
        
        # checks the compatibility of the graph
        if not tf.math.reduce_all(tf.equal(tf.sparse.sparse_dense_matmul(graph.adjacency_matrix, tf.transpose([tf.ones(graph.order(), dtype=tf.int32)])),self.degree*tf.transpose([tf.ones(graph.order(), dtype=tf.int32)]))):
            raise TypeError('The provided graph is not a ' + str(self.degree) + '-regular graph, the rule cannot be applied.')

        # computes the configuration vector
        configurations = tf.sparse.sparse_dense_matmul(graph.adjacency_matrix, graph.state_vector).numpy().transpose().squeeze() + 4*graph.state_vector.numpy().transpose().squeeze()

        # importing the list of binary digits
        rule = self.binary_digits()[::-1]

        # computes an updated state vector and a division vector
        new_state_vector = [rule[c] for c in configurations]
        division_vector =  [rule[c+2*(self.degree+1)] for c in configurations]

        while 1 in division_vector:
            i = division_vector.index(1)
            dim = len(division_vector)

            # updates the state vector
            for j in range(self.degree-1):
                new_state_vector.insert(i, new_state_vector[i])

            # updates the division vector
            division_vector[i]=0
            for j in range(self.degree-1):
                division_vector.insert(i, 0)

            # updates the adjacency matrix
            line_indices = tf.sparse.slice(graph.adjacency_matrix, [i,0], [1,dim]).indices.numpy()[:,1]
            new_lines = tf.SparseTensor(indices=[[0,line_indices[0]]], values=[1], dense_shape=[self.degree,dim])
            for k in range(1, self.degree):
                new_lines = tf.sparse.add(new_lines, tf.SparseTensor(indices=[[k,line_indices[k]]], values=[1], dense_shape=[self.degree,dim]))
            graph.adjacency_matrix = tf.sparse.concat(0,
                [
                    tf.sparse.slice(graph.adjacency_matrix, [0,0], [i,dim]),
                    new_lines,
                    tf.sparse.slice(graph.adjacency_matrix, [i+1,0], [dim-i-1,dim])
                ]
            )
            column_indices = tf.sparse.slice(graph.adjacency_matrix, [0,i], [dim+(self.degree-1),1]).indices.numpy()[:,0]
            new_columns = tf.SparseTensor(indices=[[column_indices[0],0]], values=[1], dense_shape=[dim+(self.degree-1),self.degree])
            for k in range(1, self.degree):
                new_columns = tf.sparse.add(new_columns, tf.SparseTensor(indices=[[column_indices[k],k]], values=[1], dense_shape=[dim+(self.degree-1),self.degree]))
            graph.adjacency_matrix = tf.sparse.concat(1,
                [
                    tf.sparse.slice(graph.adjacency_matrix, [0,0], [dim+(self.degree-1),i]),
                    new_columns,
                    tf.sparse.slice(graph.adjacency_matrix, [0,i+1], [dim+(self.degree-1),dim-i-1])
                ]
            )
            
            # adding the junction submatrix
            graph.adjacency_matrix = tf.sparse.add(graph.adjacency_matrix, tf.SparseTensor(indices=np.array([[m,n] for m,n in np.ndindex((self.degree,self.degree)) if m!=n])+i, values=[1 for n in range(self.degree*(self.degree-1))], dense_shape=[dim+(self.degree-1), dim+(self.degree-1)]))

        # updates the state vector
        graph.state_vector = tf.convert_to_tensor(np.array([new_state_vector]).T, dtype=tf.int32)

        # outputs the new graph
        return graph

    def jump(self, graph, n):
        for i in range(n):
            self(graph)
        return graph

    #--------------- RULE PLOT ---------------#
    def plot(self):
        # importing the list of binary digits
        rule = self.binary_digits()[::-1]

        # figure initialization
        fig, ax = plt.subplots()
        
        # layout
        sx, sy = 40, 20 # spacing
        vr = 1 # vertex radius
        el = 6 # edge length
        dst = 3.4*el # distance between config and result
        scale = 10 # output scale

        for c in range(2*self.degree+2):
            # translation
            tx = sx*(c-(self.degree+1)*(c//(self.degree+1)))
            ty = 18 if c<self.degree+1 else 0

            # adds central vertex
            x, y = tx, ty
            circle = plt.Circle((x,y), vr, facecolor='purple' if c<self.degree+1 else 'orange', edgecolor='black')
            ax.add_patch(circle) # configuration
            if rule[4*self.degree+3-c]==0: # result without division
                circle = plt.Circle((x+dst,y), vr, facecolor='purple' if rule[2*self.degree+1-c]==1 else 'orange', edgecolor='black')
                ax.add_patch(circle)
            else: # result with division
                for k in range(self.degree):
                    x, y = tx+el*np.cos(-np.pi/2+(k*2*np.pi)/self.degree), ty+el*np.sin(-np.pi/2+(k*2*np.pi)/self.degree)
                    circle = plt.Circle(((tx+x)/2+dst,(ty+y)/2), vr, facecolor='purple' if rule[2*self.degree+1-c]==1 else 'orange', edgecolor='black')
                    ax.add_patch(circle)
                    for l in range(k):
                        x_, y_ = tx+el*np.cos(-np.pi/2+(l*2*np.pi)/self.degree), ty+el*np.sin(-np.pi/2+(l*2*np.pi)/self.degree)
                        line = plt.Line2D([(tx+x)/2+dst, (tx+x_)/2+dst], [(ty+y)/2, (ty+y_)/2], color='black', zorder=-1)
                        ax.add_line(line)  

            # adds arrow
            x, y = tx+el+2*vr, ty  # start
            dx, dy = dst-2*(el+2*vr), 0 # end
            width = 3
            arrow = mpatches.Arrow(x, y, dx, dy, width, facecolor='black')
            ax.add_patch(arrow)

            # adds edges and outer vertices one-by-one
            for k in range(self.degree):
                color = 'orange' if k<(c-(self.degree+1)*(c//(self.degree+1))) else 'purple'
                x, y = tx+el*np.cos(-np.pi/2+(k*2*np.pi)/self.degree), ty+el*np.sin(-np.pi/2+(k*2*np.pi)/self.degree)
                # initial configuration
                line = plt.Line2D([tx,x], [ty, y], color='black', zorder=-1)
                circle = plt.Circle((x,y), vr, facecolor=color, edgecolor='black')
                ax.add_line(line)
                ax.add_patch(circle)
                # result at t+1
                if rule[4*self.degree+3-c]==0: # undivided
                    line = plt.Line2D([tx+dst,x+dst], [ty, y], color='black', zorder=-1)
                    ax.add_line(line)
                else: # divided
                    line = plt.Line2D([(tx+x)/2+dst,x+dst], [(ty+y)/2, y], color='black', zorder=-1)
                    ax.add_line(line)

        plt.axis('equal') # scale x and y equaly
        plt.axis('off') # don't display axes
        fig.set_size_inches(scale*self.degree, scale)
        return plt.show()