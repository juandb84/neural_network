import numpy as np

class net:
    def __init__(self,nX,arch):
        self.W = [[list(np.random.normal(0,2,nX)) for j in range(arch[0])]]
        self.bias = [list(np.random.normal(0,2,arch[0]))]
        #self.W = [[list(np.ones(nX)) for j in range(arch[0])]]
        #self.bias = [list(np.ones(arch[0]))]
        for l in range(1,len(arch)):
            self.W.append([list(np.random.normal(0,2,arch[l-1])) for j in range(arch[l])])
            self.bias.append(list(np.random.normal(0,2,arch[l])))
            #self.W.append([list(np.ones(arch[l-1])) for j in range(arch[l])])
            #self.bias.append(list(np.ones(arch[l])))
        self.a = []
        self.delta = [[] for i in range(len(arch))]
        self.dc_dw = [[] for i in range(len(arch))]
        self.dc_db = [[] for i in range(len(arch))]
        self.nX = nX
        self.arch = arch
        self.Y = [0]*arch[-1]
        
        
    def compute_output(self,X):
        if len(X) != self.nX:
            raise Exception('el tamaño de X no coincide con el tamaño del input ingresado al inicializar la red')
 
        self.a =[[self.f(self.bias[0][j]+np.dot(X,self.W[0][j])) for j in range(self.arch[0])]]

        for l in range(1,len(self.arch)):
            self.a.append([self.f(self.bias[l][j]+np.dot(self.a[l-1],self.W[l][j])) for j in range(self.arch[l])])
        self.Y = self.a[-1]
    def compute_delta(self,X,Y):
        l=-1
        self.delta[l] = [(self.a[l][j]-Y[j])*self.df(self.bias[l][j]+np.dot(self.a[l-1],self.W[l][j])) for j in range(self.arch[l])]
        l-=1
        while np.abs(l) < len(self.arch):
            WT=self.transpose_list(self.W[l+1])
            self.delta[l] = [np.dot(WT[j] , self.delta[l+1])*self.df(self.bias[l][j]+np.dot(self.a[l-1],self.W[l][j])) for j in range(self.arch[l])]
            l-=1
        l = 0
        WT=self.transpose_list(self.W[l+1])
        self.delta[l] = [np.dot(WT[j] , self.delta[l+1])*self.df(self.bias[l][j]+np.dot(X,self.W[l][j])) for j in range(self.arch[l])]

    def compute_cost_derivatives(self,X):
        l = 0
        self.dc_dw[l] = [[X[k]*self.delta[l][j] for k in range(len(X))] for j in range(self.arch[l])]

        for l in range(1,len(self.arch)):
            self.dc_dw[l] = [[self.a[l-1][k]*self.delta[l][j] for k in range(self.arch[l-1])] for j in range(self.arch[l])]
    
    def recalculate_w(self,alpha):
        for l in range(len(self.W)):
            for j in range(len(self.W[l])):
                self.bias[l][j] = self.bias[l][j] - alpha * self.delta[l][j] 
                for k in range(len(self.W[l][j])):
                    self.W[l][j][k] = self.W[l][j][k] - alpha * self.dc_dw[l][j][k] 
                
    def train_net(self,X,Y,alpha,n_steps):
        # This method trains the network with a learning rate alpha
        # n_steps number of steps for each data using gradient descending method
        for n in range(n_steps):
            self.compute_output(X)
            self.compute_delta(X,Y)
            self.compute_cost_derivatives(X)
            self.recalculate_w(alpha)
        
    def f(self,x):    
        return 1/(1+np.exp(-x))
                
    def df(self,x):
        return self.f(x)*(1-self.f(x))
    
    def transpose_list(self,a):
        return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]
    


