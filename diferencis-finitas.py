import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

'''
Se define la función pxt(m,n,mat,params) donde:
    * m es la posición del arreglo que equivale al desplazamiento en 'x'
    * n es la posición del arreglo que equivale al desplazamiento en 't'
    * mat es la matriz de valores aproximados en la iteración actual
    * params es un dicionario con los parámetros que recibe la función (en este caso son D y delta)
'''
def T(m,n,o,mat,params):
    a = params['alpha']
    densidad = params['densidad']
    Cv = params['Cv']
    delta = params['delta']
    #Bordes
    if m == 0 and n != 0 and n != int(params['Ly']//delta+1):
        pass
    elif n == 0 and m != 0 and m != int(params['Lx']//delta+1):
        pass
    elif m == int(params['Lx']//delta+1) and n != 0 and n != int(params['Ly']//delta+1):
        pass
    elif n == int(params['Ly']//delta+1) and m != 0 and m != int(params['Lx']//delta+1):
        pass
    #Esquinas
    elif m == 0 and n == 0:
        pass
    elif m == 0 and n == int(params['Ly']//delta+1):
        pass
    elif m == int(params['Lx']//delta+1) and n == 0:
        pass
    elif m == int(params['Lx']//delta+1) and n == int(params['Ly']//delta+1):
        pass
    #Resto
    else:
        return (a*(mat[o,n,m+1]+mat[o,n,m-1]+mat[o,n+1,m]+mat[o,n-1,m])-(2*a-delta*densidad*Cv)*mat[o,n,m])/(delta*densidad*Cv)
    
'''
Se define la función GaussSeidel(mat, func, prec, maxIter, params) donde:
    * mat es la matriz de valores aproximados en la iteración actual
    * func es la función en la cual se calcula el resultado de cada punto de la malla
    * prec es la precisión que se desea alcanzar
    * maxIter es el número máximo de iteraciones antes de parar la ejecución
    * params es un dicionario con los parámetros que recibe la función (en este caso son D y delta)
'''
def GaussSeidel(mat, func, prec, maxIter, params):
    #Se extraen las dimensiones del arreglo
    s = mat.shape
    
    #Se inicia el contador de iteraciones
    iters = 0
    
    #Se inicia la variable que guarda si se llegó a la precisión máxima
    maxPrec = False
    
    #Guarda la precisión para retornarla al final de la ejecución
    difFinal = 0
    # Se inicia del ciclo de iteraciones para realizar las aproximaciones
    # El ciclo para si se llega a la precisión máxima o si se llega al número máximo de iteraciones
    while (iters < maxIter and not maxPrec):
        if(iters%10 == 0):
            print(iters)
        #Se agrega 1 al contador de iteraciones
        iters += 1
        
        for m in range(1,s[2]-1):
            for n in range(1,s[1]-1):
                for o in range(0,s[0]-1):
                    #Valor anterior de la aproximación
                    f0 = mat[o+1,n,m]
                    
                    #Se calcula el nuevo valor para el punto actual
                    mat[o+1,n,m] = func(m,n,o,mat,params)
                    
                    #Se calcula la diferencia entre el valor nuevo y el anterior
                    dif = np.abs((mat[o+1,n,m] - f0)/mat[o+1,n,m])
                    if (dif < prec):
                        difFinal = dif
                        maxPrec = True

    #Se retorna como resultado la matriz con las aproximaciones, la precisión alcanzada y el número de iteraciones
    return (mat, difFinal, iters)

#Inicio del programa

#Se definen los parámetros para el cálculo que se desea realizar
params = {'delta': 0.2, 'Lx': 5, 'Ly':5, 'Lt': 100, 'densidad': 10, 'Cv': 10, 'alpha':10}

#Se define la malla de puntos y se calcula delta, que el la longitud dividida sobre la cantidad de puntos en la malla
x = np.linspace(0, params['Lx'], int(params['Lx']//params['delta']+1))
y = np.linspace(0, params['Ly'], int(params['Ly']//params['delta']+1))
# t = np.linspace(0, params['Lx'], puntosmalla)
X, Y = np.meshgrid(x, y)

#Se define la matriz inicial
Ti = np.zeros((int(params['Lt']//params['delta']+1),int(params['Ly']//params['delta']+1), int(params['Lx']//params['delta']+1)), float)

#Se introducen las condiciones iniciales en la matriz
for i in range(0, int(params['Lx']//params['delta']+1)):
    Ti[0,0,i] = 10000
    # pxti[puntosmalla-1,i] = 0
    # pxti[i,0] = params['A']*np.exp(-((params['delta']*i - params['x0'])**2)/params['l'])

#Se realiza el cálculo
(Tf,prec,iters) = GaussSeidel(Ti, T, 10**-20, 100, params)

#Se notifica al usuario sobre la precisión y el número de iteraciones
print('Precisión alcanzada:{0}\n Número de iteraciones:{1}'.format(prec,iters))

#Se grafican los resultados
ax = plt.axes(projection='3d')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('T (K)')
global grafico
grafico = ax.plot_surface(X, Y, Ti[0,:,:], rstride=1, cstride=1,
            cmap='cividis', edgecolor='none')
ax.set_title('Aproximacion Difusión en una dimensión')

axcolor = 'lightgoldenrodyellow'
ax.margins(x=0)
axTiempo = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
slider_tiempo = Slider(
    ax=axTiempo,
    label='Tiempo [s]',
    valmin=0,
    valmax=params['Lt'],
    valinit=0,
    valstep=params['delta']
)

def cambiarTiempo(tiempo):
    global grafico
    grafico.remove()
    grafico = ax.plot_surface(X, Y, Ti[int(tiempo//params['delta']),:,:], rstride=1, cstride=1,
            cmap='cividis', edgecolor='none')

slider_tiempo.on_changed(cambiarTiempo)
plt.show()