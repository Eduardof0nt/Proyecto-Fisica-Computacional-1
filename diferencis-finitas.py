import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

#Para mayor precisión
np.finfo(np.float128)
#Para excepciones
np.seterr(all='raise')

'''
Se define la función T(m,n,o,mat,params) donde:
    * m es la posición del arreglo que equivale al desplazamiento en 'x'
    * n es la posición del arreglo que equivale al desplazamiento en 'y'
    * o es la posición del arreglo que equivale al desplazamiento en 't'
    * mat es la matriz de valores aproximados en la iteración actual
    * params es un dicionario con los parámetros que recibe la función (en este caso son alpha y delta)
'''
def T(m,n,o,mat,params):
    a = params['alpha']
    delta = params['delta']
    return (a/delta)*(mat[o,n,m+1]+mat[o,n,m-1]+mat[o,n+1,m]+mat[o,n-1,m]-4*mat[o,n,m])+mat[o,n,m]
    
'''
Se define la función GaussSeidel(mat, func, prec, maxIter, params) donde:
    * mat es la matriz de valores aproximados en la iteración actual
    * func es la función en la cual se calcula el resultado de cada punto de la malla
    * prec es la precisión que se desea alcanzar
    * maxIter es el número máximo de iteraciones antes de parar la ejecución
    * params es un dicionario con los parámetros que recibe la función
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
        if(iters%1 == 0):
            print(iters)
        #Se agrega 1 al contador de iteraciones
        iters += 1
        
        #Se itera sobre los elementos del arreglo para cada punto (x,y,t).
        #En la matriz se debe ingresar el índice al revés de (x,y,t), es decir, se ingresa como [o,n,m], los cuales corresponden a (t,y,x)
        #Las iteraciones de hacen desde el tiempo inicial (0) hasta el penúltimo tiempo y en "x" y "y" desde 1 hata el penúltimo elemento, se excluyen los contronos.
        #En el caso de "t" se hace de esta forma debido a que en cada iteración se utiliza el estado en el tiempo actual para calcular el estado en el siguiente paso de tiempo (o+1)
        for o in range(0,s[0]-1):
            for n in range(1,s[1]-1):
                for m in range(1,s[2]-1):
                    #Valor anterior de la aproximación (valor en la iteración anterior)
                    f0 = mat[o+1,n,m]

                    #Se calcula el nuevo valor para el punto actual utilizando la función ingresada
                    mat[o+1,n,m] = func(m,n,o,mat,params)
                    
                    #Se calcula la diferencia entre el valor nuevo y el anterior para calcular la presición y si se llegó a la deseada terminar el ciclo
                    try:
                        dif = np.abs((mat[o+1,n,m] - f0)/mat[o+1,n,m])
                    except:
                        dif = 0
                    if (dif < prec):
                        difFinal = dif
                        maxPrec = True
            #Condición de contorno para que no haya transfernecia de calor en los bordes, o sea para bordes aislados del ambiente        
            # mat[o+1,0,:] = mat[o+1,1,:]
            # mat[o+1,-1,:] = mat[o+1,-2,:]
            # mat[o+1,:,0] = mat[o+1,:,1]
            # mat[o+1,:,-1] = mat[o+1,:,-2]
    #Se retorna como resultado la matriz con las aproximaciones, la precisión alcanzada y el número de iteraciones
    return (mat, difFinal, iters)

#----------Inicio del programa----------

#----------Definiciones Iniciales del Programa----------
#Se definen los parámetros para el cálculo que se desea realizar
params = {'delta': 0.05, 'Lx': 1, 'Ly':1, 'Lt': 10**2, 'alpha': 149*10**-6, 'A': 100}

#Se define la malla de puntos y se calcula delta, que el la longitud dividida sobre la cantidad de puntos en la malla
x = np.linspace(0, params['Lx'], int(params['Lx']//params['delta']+1))
y = np.linspace(0, params['Ly'], int(params['Ly']//params['delta']+1))
# t = np.linspace(0, params['Lx'], puntosmalla)
X, Y = np.meshgrid(x, y)

#Se define la matriz inicial (si se multiplica la matriz por otro término diferente de 0 se puede fijar una temperatura inicial de la placa diferente)
Ti = 0*np.ones((int(params['Lt']//params['delta']+1),int(params['Ly']//params['delta']+1), int(params['Lx']//params['delta']+1)),dtype=np.float128)

#----------Condiciones Iniciales y de Controno----------

#Se definen los parámetros del centro de la placa para las condiciones iniciales (opcional, solo se usa si se desean las condiciones iniciales planteadas o similares)
x0 = int((params['Lx']//params['delta']+1)/2)
y0 = int((params['Ly']//params['delta']+1)/2)

#Se introducen las condiciones iniciales en la matriz

#Condiciones iniciales 1, "el dedo caliente"
# for i in range(0, int(params['Lx']//params['delta']+1)):
#     for j in range(0, int(params['Ly']//params['delta']+1)):
#         #Se usan excepciones para evitar los overflows por valores que se salen del rango del tipo de variable numérica
#         try:
#             Ti[0,i,j] = (params['A']**2)*np.exp((-(i-x0)**2)/(10*params['Lx'])-((j-y0)**2)/(10*params['Ly'])) + Ti[0,i,j] #Para el caso del dedo caliente en el centro de la placa
#         except:
#             Ti[0,i,j] = 0

#Condiciones iniciales 2, la muralla de temperatura
for i in range(0, int(params['Lx']//params['delta']+1)):
    #Se usan excepciones para evitar los overflows por valores que se salen del rango del tipo de variable numérica
    try:
        Ti[0,2:4,i] = (params['A'])*np.exp((-(i-x0)**2)/(100*params['Lx']))#Para una condición de temperatura inicial bidimencional.
    except:
        Ti[0,2:4,i] = 0



#Se introducen las condiciones de controno (opcional, solo se usa en caso de que la frontera sea un reservorio de calor a una temperatura fija)
Ti[:,:,0] = 0
Ti[:,:,-1] = 0
Ti[:,0,:] = 0
Ti[:,-1,:] = 0

#------------------Simulación-----------------

#Se realiza el cálculo
(Tf,prec,iters) = GaussSeidel(Ti.copy(), T, 10**-2, 100, params)

#Se notifica al usuario sobre la precisión y el número de iteraciones
print('Precisión alcanzada:{0}\n Número de iteraciones:{1}'.format(np.format_float_scientific(prec),iters))


#-----------------Gráfico de los Resultados de la Simulación------------

#Se indican los parámetros para la graficación de los resultados
ax = plt.axes(projection='3d')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('T (°C)')
ax.set_zlim(0,200)
ax.set_title('Aproximacion Difusión de Calor en una Placa 2D')

#Se define la variable "grafico" como global para ser utilizada en otra función luego
global grafico

#Se grafican los resultados del instante 0, las condiciones iniciales
grafico = ax.plot_surface(X, Y, Tf[0,:,:], rstride=1, cstride=1,
            cmap='cividis', edgecolor='none')

#Definición del slider para cambiar el instante de tiempo que se muestra
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

#Se define la función "cambiarTiempo" para cambiar el instante de tiempo que se muestra en la gráfica
def cambiarTiempo(tiempo):
    #Primero se obtiene la variable global "grafico" definida anteriormente y se remueve su contenido
    global grafico
    grafico.remove()
    
    #Se vuelve a graficar sobre la variable "grafico", pero en el instante de tiempo indicado por el slider (la posición a la cual lo movió el usuario)
    grafico = ax.plot_surface(X, Y, Tf[int(tiempo//params['delta']),:,:], rstride=1, cstride=1,
            cmap='cividis', edgecolor='none')

#Se le ingresa la función definida anteriormente como función a llamar en el momento en el que se cambie el slider (callback function)
slider_tiempo.on_changed(cambiarTiempo)

#Se muestra la gráfica realizada
plt.show()