from red import net
import numpy as np



N = 1000 #tamaño de la muestra
X = [list(np.random.normal(0,1,6)) for i in range(N)]
Y=[[X[i][0]**2+X[i][1]*X[i][2]+X[i][4]/(1+X[i][5])+np.random.normal(0,2)] for i in range(N)]





red = net([8,1])
red.generate_net(X[0])  #esto genera todas las c apas, que van ocn input y outpt, por eso es necesario apsarle un vector de inputs, pero puede ser un vector de cualquier cosa q tenga el tamñao del input

alpha = 0.1 #learning rate

#train the network with model data
for i in range(N):
    red.train_net(X[i],Y[i],alpha)

