from red import net
import numpy as np
import matplotlib.pyplot as plt


N = 200000 #tamaño de la muestra
#X = [list(np.random.normal(0,1,6)) for i in range(N)]
#Y=[[X[i][0]**2+X[i][1]*X[i][2]+X[i][4]/(1+X[i][5])+np.random.normal(0,2)] for i in range(N)]

X=[[0,0],[0,1],[1,0],[1,1]]
y=[[0.],[1.],[1.],[0.]]




red = net(len(X[0]),[2,1])

alpha = 0.5 #learning rate

v_error=[]

#train the network with model data
for k in range(N):  # itera sobre todos los datos
    for i in range(len(X)):   #itera sobre cada dato
        red.train_net(X[i],y[i],alpha,2)  #lo corre dos veces por dato
    error = 0
    for xx,yy in zip(X,y):
        red.compute_output(xx)
        error+= (red.Y[0]-yy[0])**2
    v_error.append(error)

for x in X:
    red.compute_output(x)
    print(red.Y)
    
plt.plot(v_error)