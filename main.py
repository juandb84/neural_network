from red import net
import numpy as np



N = 1000 #tamaño de la muestra
X = [list(np.random.normal(0,1,6)) for i in range(N)]
Y=[[X[i][0]**2+X[i][1]*X[i][2]+X[i][4]/(1+X[i][5])+np.random.normal(0,2)] for i in range(N)]





red = net(len(X[0]),[8,1])

alpha = 0.1 #learning rate

#train the network with model data
for i in range(N):
    red.train_net(X[i],Y[i],alpha,1)

