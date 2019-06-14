# -*- coding: utf-8 -*-


import numpy as np

            
class layer(object):
    def __init__(self, n_layer, inputs):
        self.n_layer = n_layer #number of neurons of the layer
        self.inputs = inputs   #list of inputs from previous layer
        self.output = []
        self.W = [list(np.random.normal(0,2,len(inputs))) for i in range(n_layer)]
        self.bias = list(np.random.normal(0,2,n_layer))
    
    def generate_out(self):
        for i in range(self.n_layer):
            Z = self.bias[i] + np.dot(self.W[i],self.inputs)
            Z = 1 / (1 + np.exp(-Z))
            self.output.append(Z)

class net(object):
    def __init__(self, arch, X, Y):
        self.arch = arch #lista q me dice cuantas neuronas tiene cada capa
        self.X = X  #El input de la red
        self.layer = []
        self.delta =  [0]*len(self.arch)
        self.Y = Y
        self.dcdw = []
        self.dcdb = []
        
    def generate_net(self):
        self.layer.append(layer(self.arch[0],self.X))
        self.layer[0].generate_out()
        for i in range(1,len(self.arch)):
            self.layer.append(layer(self.arch[i], self.layer[i-1].output))
            self.layer[i].generate_out()
    def compute_delta(self):
        k=len(self.delta)-1
        self.delta[k] = [(self.layer[k].output[i]-self.Y[i])*self.layer[k].output[i]*(1-self.layer\
        [k].output[i]) for i in range(len(self.layer[k].output))]
        while k >= 0:
        
            k-=1    
            self.delta[k] = [np.dot(list(np.array(self.layer[k+1].W)[:,i]),self.delta[k+1])\
            *self.layer[k].output[i]*(1-self.layer[k].output[i])\
            for i in range(self.layer[k].n_layer)]
    def compute_cost_derivatives(self):
        

                

X=[1,1,1,1,1]
Y=[1,1]

        
red = net([3,4,2],X,Y)
red.generate_net()
red.compute_delta()
