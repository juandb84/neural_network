# -*- coding: utf-8 -*-


import numpy as np

            
class layer(object):
    def __init__(self, n_layer, inputs):
        self.n_layer = n_layer #number of neurons of the layer
        self.inputs = inputs   #list of inputs from previous layer
        self.output = []
        self.W = [list(np.random.normal(0,2,len(inputs))) for i in range(n_layer)]
        self.bias = list(np.random.normal(0,2,n_layer))
    
    def generate_out(self,X):
        self.output = []
        for i in range(self.n_layer):
            Z = self.bias[i] + np.dot(self.W[i],X)
            Z = 1 / (1 + np.exp(-Z))
            self.output.append(Z)

class net(object):
    def __init__(self, arch):
        self.arch = arch #lista q me dice cuantas neuronas tiene cada capa
 
        
        self.layer = []
        self.delta =  [0]*len(self.arch)
        
        self.dcdw = []
        self.dcdb = []
        self.costo = 0
        
    def generate_net(self,X):
        self.layer.append(layer(self.arch[0],X))
        self.layer[0].generate_out(X)
        for i in range(1,len(self.arch)):
            self.layer.append(layer(self.arch[i], self.layer[i-1].output))
            self.layer[i].generate_out(self.layer[i-1].output)
    
    def compute_delta(self,Y):
        k=len(self.delta)-1
        self.delta[k] = [(self.layer[k].output[i]-Y[i])*self.layer[k].output[i]*(1-self.layer\
        [k].output[i]) for i in range(len(self.layer[k].output))]
        while k >= 0:
        
            k-=1    
            self.delta[k] = [np.dot(list(np.array(self.layer[k+1].W)[:,i]),self.delta[k+1])\
            *self.layer[k].output[i]*(1-self.layer[k].output[i])\
            for i in range(self.layer[k].n_layer)]
    
    def compute_cost_derivatives(self,X):
        for i in range(len(self.arch)):
            if i==0:
                a_k_lm1 = X
            else:
                a_k_lm1 = self.layer[i-1].output
            d_j_l = self.delta[i]
            
            self.dcdb.append(self.delta[i])
            self.dcdw.append([[d*a for a in a_k_lm1] for d in d_j_l])
            
    def recalculate_W_b(self,alpha):
        for j in range(len(self.arch)):
            W = self.layer[j].W
            b = self.layer[j].bias
            self.layer[j].W = [[x1[i]-alpha*x2[i] for i in range(len(x1))] for x1,x2 in zip(W,self.dcdw[j])]
            self.layer[j].bias = [x1-alpha*x2 for x1,x2 in zip(b,self.dcdb[j])]
            
    
    def compute_out_all_net(self,X):
        self.layer[0].generate_out(X)
        for i in range(1,len(self.arch)):
            self.layer[i].generate_out(self.layer[i-1].output)
    
    def train_net(self,X,Y,alpha):
        self.compute_delta(Y)
        self.compute_cost_derivatives(X)
        self.recalculate_W_b(alpha)
        self.compute_out_all_net(X)
        self.costo = sum([(Y[i]-self.layer[-1].output[i])**2 for i in range(len(Y))])
        
            
        
        

                


