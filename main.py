from red import net
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

N = 1000 #nro de iteraciones de los datos en el entrenamiento
M=15  #tamaño del input
X = [list(np.random.normal(0,1,4)) for i in range(M)]
y=[[(X[i][0]*X[i][3]-X[i][1]*X[i][2])] for i in range(M)]


## Compuerta XOR
#X=[[0,0],[0,1],[1,0],[1,1]]
#y=[[0.],[1.],[1.],[0.]]


#include quadratic terms in the model
for k in range(len(X)):
    for i, j in itertools.product(range(len(X[k])), range(len(X[k]))):
        X[k].append(X[k][i]*X[k][j])

N_iterations = 10

for u in range(N_iterations):
    red = net(len(X[0]),[2,1])

    alpha = 0.3 #learning rate

    v_error=[]


#iterate several times to get different learning curves



    #train the network with model data
    percentage=1
    for k in range(N):  # itera sobre todos los datos
        for i in range(len(X)):   #itera sobre cada dato
            red.train_net(X[i],y[i],alpha,5)  #lo corre dos veces por dato
        error = 0
        for xx,yy in zip(X,y):
            red.compute_output(xx)
            error+= (red.Y[0]-yy[0])**2
        v_error.append(error)    
        if percentage < 100*(k+1)/N:
            percentage += 1
            print('Traning advance: {} % iteration {} from {}'.format(np.ceil(100*k/N), u +1 , N_iterations))

    for x in X:
        red.compute_output(x)
        print(red.Y)
        
    plt.plot(v_error)

    plt.xlabel('n iteraciones')
    plt.ylabel('error cuadratico')

    plt.title('Error de la red en funcion del nro de iteraciones')


plt.show()