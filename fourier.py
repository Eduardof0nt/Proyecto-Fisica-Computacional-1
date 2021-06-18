# Se importan las bibliotecas necesarias.
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tkinter import Tk, Scale, CENTER, Button, Label, mainloop
from scipy import integrate
import animatplot as amp

# Variables estipuladas para el cálculo.
s = 1.5
alpha = 0.5
A = 2


def CalculoDifusiónInicial (variable, inicioVariable):
    """
    Se realiza el cálculo de la difusión cuando el valor del tiempo es 0 para las posiciones dadas.
    
    Parámetros de la función:
    ------------------------
    variable: Posiciones en un de los ejes dimensionales de la placa.
    inicioVariable: Punto en el que se coloca el calor. Valor x0 o y0 de la ecuación.
    
    Salida de la función:
    ---------------------
    Valor de la difusión inicial para las posiciónes definidas.
    """

    valorD0 = A * np.exp (-(variable - inicioVariable) ** 2 / s)

    return valorD0


def CalculoCoeficienteFourier (nContador, lado, variable, inicioVariable1, cómoCalor):
    """
    Se realiza el cálculo del coeficiente de la serie de Fourier.
    
    Parámetros de la función:
    ------------------------
    nContador= Valor del contador o n para la serie de Fourier.
    lado: Tamaño de lado que se está evaluando.
    variable: Posiciones en un de los ejes dimensionales de la placa.
    inicioVariable: Punto en el que se coloca el calor. Valor x0 o y0 de la ecuación.
    Salida de la función:
    ---------------------
    Valor del coeficiente de la serie de fourier para una iteracion n específica.
    """ 

    if cómoCalor == 0:
        funciónIntegrar = lambda variable: 2/lado * A * np.exp (-variable**2 / (7.5*s)) * np.sin (nContador * np.pi * variable / lado)

        coeficienteFourier =  integrate.quad (funciónIntegrar, 0, lado) [0]

    else:
        funciónIntegrar = lambda variable: 2/lado * CalculoDifusiónInicial (variable, inicioVariable1) * np.sin (nContador * np.pi * variable / lado)

        coeficienteFourier =  integrate.quad (funciónIntegrar, 0, lado) [0]

    return coeficienteFourier


def CalculoDifusión (t, x, x0, tamañoLadoX, y, y0, tamañoLadoY, númeroTérminosN, númeroTérminosM, cómoCalor):
    """
    Realiza el cálculo de la ecuación de difusión en dos dimensioes a través del tiempo por medio del método de Series de Fourier.
    Parámetros de la función:
    ------------------------
    t = Tiempo en el que se realiza el cálculo.
    x y y: Valores correspondientes a los ejes dimensionales de la placa.
    tamañoLadoX y tamañoLadoY: Dimensiones de la placa.
    x0 y y0: Puntos donde se aplica el calor a la placa.
    númeroTérminosN y númeroTérminosM: Cantidad de términos con los que se realizará el cálculo por series de Fourier.
    cómoCalor: Variable que indica la forma en que se está aplicando el calor a la placa.
    
    Salida de la función:
    ---------------------
    Valor aproximado de la ecuación de difusión en dos dimensiones.
    """

    potencialAproximado = 0

    if cómoCalor == 0:
        for nContador in range (1, númeroTérminosN):
            for mContador in range (1, númeroTérminosM):

                coeficienteCn = CalculoCoeficienteFourier (nContador, tamañoLadoX, x, x0, cómoCalor)
                
                funciónIntegrar = lambda y: 2/tamañoLadoY * np.sin (nContador * np.pi * y / tamañoLadoY)
                coeficienteCm = integrate.quad (funciónIntegrar, 0, tamañoLadoY) [0]

                potencialAproximado += coeficienteCn*np.sin(nContador*np.pi*x/tamañoLadoX)*np.exp(-alpha*np.pi**2*t*nContador**2/tamañoLadoX**2) \
                * coeficienteCm*np.sin(mContador*np.pi*y/tamañoLadoY)*np.exp(-alpha*np.pi**2*t*mContador**2/tamañoLadoY**2)

        return potencialAproximado

    else:
        for nContador in range (1, númeroTérminosN):
            for mContador in range (1, númeroTérminosM):

                coeficienteCn = CalculoCoeficienteFourier (nContador, tamañoLadoX, x, x0, cómoCalor)
                coeficienteCm = CalculoCoeficienteFourier (mContador, tamañoLadoY, y, y0, cómoCalor)

                potencialAproximado += coeficienteCn*np.sin(nContador*np.pi*x/tamañoLadoX)*np.exp(-alpha*np.pi**2*t*nContador**2/tamañoLadoX**2) \
                * coeficienteCm*np.sin(mContador*np.pi*y/tamañoLadoY)*np.exp(-alpha*np.pi**2*t*mContador**2/tamañoLadoY**2)

        return potencialAproximado


def AbrirGrafica3D (tamañoLadoX, x0, númeroTérminosN, tamañoLadoY, y0, númeroTérminosM, cómoCalor):
    """
    Se genera una nueva ventana en la que se despliegan los resultados gráficos animados en 3D para la difusión.
    intensidad del cálculo de difusión establecido.
    
    Parámetros de la función:
    ------------------------
    tamañoLadoX y tamañoLadoY: Dimensiones de la placa.
    x0 y y0: Puntos donde se aplica el calor a la placa.
    númeroTérminosN y númeroTérminosM: Cantidad de términos con los que se realizará el cálculo por series de Fourier.
    cómoCalor: Variable que indica la forma en que se está aplicando el calor a la placa.
    
    Salida de la función:
    ---------------------
    Grafica en 3D animada de los cálculos realizados.
    """
   
    # Se definen los puntos en los que se va a evaluar la ecuación.
    puntosMalla = 31

    x = np.linspace (0, tamañoLadoX, puntosMalla)
    y = np.linspace (0, tamañoLadoY, puntosMalla)

    t = np.linspace (0,5,50)

    X, Y = np.meshgrid (x, y)


    fig1 = plt.figure ()
    ax = fig1.gca(projection = '3d')

    def actualizar (i):
        ax.clear ()
        difusion = CalculoDifusión(t[i], X, x0, tamañoLadoX, Y, y0, tamañoLadoY, númeroTérminosN, númeroTérminosM, cómoCalor)
        ax.plot_surface (X, Y, difusion)
        plt.title (r"Ecuación de Difusión bidimensional dependiente de"
        "\n" 
        r"la posición por método de Series de Fourier")
        ax.set_xlabel ('x (m)')
        ax.set_ylabel ('y (m)')
        ax.set_zlabel ('u (x,y,t)')
        ax.set_zlim3d([0.0, 5])
    
    plt.get_current_fig_manager().window.showMaximized ()
    ani = animation.FuncAnimation (fig1,actualizar,range (len(t)), interval = 1)

    plt.show ()


def AbrirGraficaLateralIntensidad (tamañoLadoX, x0, númeroTérminosN, tamañoLadoY, y0, númeroTérminosM, cómoCalor):
    """
    Se genera una nueva ventana en la que se despliegan los resultados gráficos, tanto una vista lateral, como una gráfica de
    intensidad del cálculo de difusión establecido.
    
    Parámetros de la función:
    ------------------------
    tamañoLadoX y tamañoLadoY: Dimensiones de la placa.
    x0 y y0: Puntos donde se aplica el calor a la placa.
    númeroTérminosN y númeroTérminosM: Cantidad de términos con los que se realizará el cálculo por series de Fourier.
    cómoCalor: Variable que indica la forma en que se está aplicando el calor a la placa.
    
    Salida de la función:
    ---------------------
    Grafica con vista lateral y gráfica de intensidad de los cálculos realizados.
    """
    
    # Se definen los puntos en los que se va a evaluar la ecuación.
    puntosMalla = 30

    # Se crea un arreglo con los valores para cada variable.
    x = np.linspace (0, tamañoLadoX, puntosMalla)
    y = np.linspace (0, tamañoLadoY, puntosMalla)
    t = np.linspace (0,5,50)

    # Se unifican las tres variables para la graficación de intensidad. 
    # Luego se unifican las variables posicionales para la graficación lateral.
    X, Y, T= np.meshgrid (x, y, t)
    X1, Y1 = np.meshgrid (x, y)

    # Se define una variable pcolormesh_data la cual contiene el resultado de la difusión.
    pcolormesh_data = CalculoDifusión (T, X, x0, tamañoLadoX, Y, y0, tamañoLadoY, númeroTérminosN, númeroTérminosM, cómoCalor)


    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Graficación de intensidad.
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')

    ax2.set_ylabel('y (m)', labelpad=-5)

    fig.suptitle(r'Evolución de la difusión '
    "\n" 
    r'a través del tiempo')
    ax2.set_title('Gráfica de Intensidad')

    pcolormesh_block = amp.blocks.Pcolormesh(X[:,:,0], Y[:,:,0], pcolormesh_data,
                                            ax=ax2, t_axis=2, vmin=0, vmax=5)
    plt.colorbar(pcolormesh_block.quad)
    timeline = amp.Timeline(t, fps=10)

    anim = amp.Animation([pcolormesh_block], timeline)


    # Graficación de vista lateral.
    def actualizar (i):
        ax1.clear ()
        difusion = CalculoDifusión(t[i], X1, x0, tamañoLadoX, Y1, y0, tamañoLadoY, númeroTérminosN, númeroTérminosM, cómoCalor)
        ax1.plot (x, difusion)
        ax1.set_title('Vista lateral')
        ax1.set_xlabel('y (m)')
        ax1.set_ylabel('u (x, y, t)')
        ax1.set_ylim([0,5])

    ani = animation.FuncAnimation (fig,actualizar,range (len(t)), interval = 1)


    plt.get_current_fig_manager().window.showMaximized ()
    plt.show()


#########################################################################################
# Sección Gráfica de la simulación.   
        

def VentanaFouierUnPunto (ventana):
    """
    Ventana en la que el usuario define los parámetros para el cálculo.
    Parámetros de la función:
    ------------------------
    ventana: Variable que cierra una ventana ya abierta.
    
    Salida de la función:
    ---------------------
    Interfaz gráfica con la que el usuario elije un método de simulación.
    """

    ventana.destroy ()
    ventanaFourUnPunto = Tk ()
    ventanaFourUnPunto.minsize (1300,600)
    ventanaFourUnPunto.config (bg='White')

    barraLadoX = Scale(ventanaFourUnPunto, from_=7, to=20, tickinterval=13, length=400, bg = 'White',
    resolution=1, showvalue=True, orient='horizontal', label="Tamaño del Lado (Lx)", cursor = "hand1")
    barraLadoX.set(10)
    tamañoLadoX = barraLadoX.get()
    barraLadoX.place (relx=0.175, rely=0.35, anchor=CENTER)

    barraX0 = Scale(ventanaFourUnPunto, from_=0, to=10, tickinterval=10,length=400, bg = 'White',
    resolution=1, showvalue=True, orient='horizontal', label="Punto Incial X", cursor = "hand1")
    barraX0.set(5)
    x0 = barraX0.get()
    barraX0.place  (relx=0.5, rely=0.35, anchor=CENTER)

    barraTérminosN = Scale(ventanaFourUnPunto, from_=2, to=20, tickinterval=18,length=400, bg = 'White',
    resolution=1, showvalue=True, orient='horizontal', label="Número de Términos N", cursor = "hand1")
    barraTérminosN.set(10)
    númeroTérminosN = barraTérminosN.get()
    barraTérminosN.place (relx=0.825, rely=0.35, anchor=CENTER)

    barraLadoY = Scale(ventanaFourUnPunto, from_=7, to=20, tickinterval=13, length=400, bg = 'White',
    resolution=1, showvalue=True, orient='horizontal', label="Tamaño del Lado (Ly)", cursor = "hand1")
    barraLadoY.set(10)
    tamañoLadoY = barraLadoY.get()
    barraLadoY.place (relx=0.175, rely=0.5, anchor=CENTER)

    barraY0 = Scale(ventanaFourUnPunto, from_=0, to=10, tickinterval=10,length=400, bg = 'White',
    resolution=1, showvalue=True, orient='horizontal', label="Punto Incial Y", cursor = "hand1")
    barraY0.set(5)
    y0 = barraY0.get()
    barraY0.place (relx=0.5, rely=0.5, anchor=CENTER)

    barraTérminosM = Scale(ventanaFourUnPunto, from_=2, to=20, tickinterval=18,length=400, bg = 'White',
    resolution=1, showvalue=True, orient='horizontal', label="Número de Términos M", cursor = "hand1")
    barraTérminosM.set(10)
    númeroTérminosM = barraTérminosM.get()
    barraTérminosM.place (relx=0.825, rely=0.5, anchor=CENTER)

    etiquetaParámetros = Label (ventanaFourUnPunto, text = ('Defina los parámetros iniciales: '))
    etiquetaParámetros.config (bg = 'white', font = ('Verdana', 12))
    etiquetaParámetros.place (relx=0.1225, rely=0.25, anchor=CENTER)

    boton3D = Button(ventanaFourUnPunto, text='Abrir Gráfica 3D', 
    command = lambda: AbrirGrafica3D (tamañoLadoX, x0, númeroTérminosN, tamañoLadoY, y0, númeroTérminosM, 1))
    boton3D.place (relx=0.42, rely=0.65, anchor=CENTER)

    botonLatInt = Button(ventanaFourUnPunto, text='Abrir Grafica de Intensidad',
    command = lambda: AbrirGraficaLateralIntensidad (tamañoLadoX, x0, númeroTérminosN, tamañoLadoY, y0, númeroTérminosM, 1))
    botonLatInt.place (relx=0.6, rely=0.65, anchor=CENTER)

    botonRegresarPrincipal = Button(ventanaFourUnPunto, text='Regresar', command = lambda: VentanaFourierPrincipal (ventanaFourUnPunto))
    botonRegresarPrincipal.place (relx=0.835, rely=0.9, anchor=CENTER) 

    ventanaFourUnPunto.title ("Cálculo de la ecuación de difusión de calor")
    ventanaFourUnPunto.mainloop()


def VentanaFourierLineal (ventana):
    """
    Ventana en la que el usuario define los parámetros para el cálculo.
    Parámetros de la función:
    ------------------------
    ventana: Variable que cierra una ventana ya abierta.
    
    Salida de la función:
    ---------------------
    Interfaz gráfica con la que el usuario elije un método de simulación.
    """

    ventana.destroy ()

    ventanaFourLineal = Tk ()
    ventanaFourLineal.minsize (1300,600)
    ventanaFourLineal.config (bg='White')

    barraLadoX = Scale(ventanaFourLineal, from_=7, to=20, tickinterval=13, length=400, bg = 'White',
    resolution=1, showvalue=True, orient='horizontal', label="Tamaño del Lado (Lx)", cursor = "hand1")
    barraLadoX.set(10)
    tamañoLadoX = barraLadoX.get()
    barraLadoX.place (relx=0.3, rely=0.35, anchor=CENTER)

    barraTérminosN = Scale(ventanaFourLineal, from_=2, to=20, tickinterval=18,length=400, bg = 'White',
    resolution=1, showvalue=True, orient='horizontal', label="Número de Términos N", cursor = "hand1")
    barraTérminosN.set(10)
    númeroTérminosN = barraTérminosN.get()
    barraTérminosN.place (relx=0.7, rely=0.35, anchor=CENTER)

    barraLadoY = Scale(ventanaFourLineal, from_=7, to=20, tickinterval=13, length=400, bg = 'White',
    resolution=1, showvalue=True, orient='horizontal', label="Tamaño del Lado (Ly)", cursor = "hand1")
    barraLadoY.set(10)
    tamañoLadoY = barraLadoY.get()
    barraLadoY.place (relx=0.3, rely=0.55, anchor=CENTER)

    barraTérminosM = Scale(ventanaFourLineal, from_=2, to=20, tickinterval=18,length=400, bg = 'White',
    resolution=1, showvalue=True, orient='horizontal', label="Número de Términos M", cursor = "hand1")
    barraTérminosM.set(10)
    númeroTérminosM = barraTérminosM.get()
    barraTérminosM.place (relx=0.7, rely=0.55, anchor=CENTER)

    etiquetaParámetros = Label (ventanaFourLineal, text = ('Defina los parámetros iniciales: '))
    etiquetaParámetros.config (bg = 'white', font = ('Verdana', 12))
    etiquetaParámetros.place (relx=0.1225, rely=0.2, anchor=CENTER)

    boton3D = Button(ventanaFourLineal, text='Abrir Gráfica 3D', 
    command = lambda: AbrirGrafica3D (tamañoLadoX, 0, númeroTérminosN, tamañoLadoY, 0, númeroTérminosM, 0))
    boton3D.place (relx=0.42, rely=0.7, anchor=CENTER)

    botonLatInt = Button(ventanaFourLineal, text='Abrir Grafica de Intensidad',
    command = lambda: AbrirGraficaLateralIntensidad (tamañoLadoX, 0, númeroTérminosN, tamañoLadoY, 0, númeroTérminosM, 0))
    botonLatInt.place (relx=0.6, rely=0.7, anchor=CENTER)

    botonRegresarPrincipal = Button(ventanaFourLineal, text='Regresar', command = lambda: VentanaFourierPrincipal (ventanaFourLineal))
    botonRegresarPrincipal.place (relx=0.835, rely=0.85, anchor=CENTER) 

    ventanaFourLineal.title ("Cálculo de la ecuación de difusión de calor")
    ventanaFourLineal.mainloop()


def VentanaFourierPrincipal (ventana):
    """
    Ventana en la que el usuario elige cómo aplicar calor a la placa.
    Parámetros de la función:
    ------------------------
    ventana: Variable que cierra una ventana ya abierta.
    
    Salida de la función:
    ---------------------
    Interfaz gráfica con la que el usuario elije un método de simulación.
    """

    ventana.destroy ()
    ventanaFourierPrincipal = Tk ()
    ventanaFourierPrincipal.minsize (800,600)
    ventanaFourierPrincipal.config (bg='White')

    etiquetaInfo = Label (ventanaFourierPrincipal, text = ('Cáculo de difusión de Calor por Series de Fourier'))
    etiquetaInfo.config (bg = 'white', font = ('Verdana', 17))
    etiquetaInfo.place (relx=0.5, rely=0.25, anchor=CENTER)

    etiquetaCalor = Label (ventanaFourierPrincipal, text = ('Cómo aplicar el calor: '))
    etiquetaCalor.config (bg = 'white', font = ('Verdana', 12))
    etiquetaCalor.place (relx=0.15, rely=0.5, anchor=CENTER)

    botonUnPunto = Button(ventanaFourierPrincipal, text='Un punto', command = lambda: VentanaFouierUnPunto (ventanaFourierPrincipal))
    botonUnPunto.place (relx=0.4, rely=0.6, anchor=CENTER)
   
    botonLineal = Button(ventanaFourierPrincipal, text='Linealmente', command = lambda: VentanaFourierLineal (ventanaFourierPrincipal))
    botonLineal.place (relx=0.6, rely=0.6, anchor=CENTER) 

    ventanaFourierPrincipal.title ("Cálculo de la ecuación de difusión de calor por Fourier")
    ventanaFourierPrincipal.mainloop()


ventana = Tk ()
VentanaFourierPrincipal (ventana)