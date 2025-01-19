import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import airy


# Fijar la semilla para reproducibilidad
# seed_value = 47  # Puedes cambiar este valor por cualquier número entero
# random.seed(seed_value)
# np.random.seed(seed_value)







# Parámetros
N = 100
dt = 0.05
g = 2.0
h = 0.00
maxel = 0
#Variables histograma
nbins = 1000
xmax= 100
xmin= 0
nsamples = 1000000/4
anchoInt = (xmax - xmin) / nbins # Cálculo del ancho del intervalo

path = np.zeros(N + 1)  # Inicializar trayectorias
prob = np.zeros(nbins)  # Inicializar histograma
pdf = np.zeros(nbins)   # Inicializar density probability function


# Obtener el primer cero de la función de Airy
x0 = 2.338107410459767  # Primer cero de Ai(x), conocido

_, Ai_prime, _, _ = airy(x0)  # Obtén el valor de Ai'(-z_n)
N_n = np.sqrt(abs(Ai_prime))    # N_n = sqrt(|Ai'(-z_n)|)

# Configurar las gráficas
fig, axes = plt.subplots(figsize=(8, 10))

# Gráfica de función de onda
ax2 = axes
ax2.set_title("Ground State Probability and Exact Solution")
ax2.set_xlabel("x")
ax2.set_ylabel("Probability")


# Inicializar gráficos
wavefunction_line, = ax2.plot([], [], color='black', linewidth=1, label="Simulated Solution")
exact_solution_line, = ax2.plot([], [], color='blue', linestyle='--', label="Exact Solution")
ax2.legend()
# Cálculo de energía
def energy(arr):
    kinetic = 0.5 * np.sum(((arr[1:] - arr[:-1]) / dt) ** 2)
    potential = g * np.sum((arr[:-1] + arr[1:]) / 2)
    return kinetic + potential

# Método para graficar la función de onda
def plotwf(prob):
    x_vals = np.arange(xmin,xmax,anchoInt) #-AnchoInt?
    y_vals = prob #* 1250.0/ 100.0
    wavefunction_line.set_data(x_vals, y_vals)
    ax2.set_xlim(min(x_vals), 10)#max(x_vals))
    ax2.set_ylim(min(y_vals), 0.4)
    fig.savefig("probPlot.pdf")

def plot_exact():
    # m=100/1250
    m=1
    x_vals = np.linspace(0,500,5000)
    airy_vals = airy((x_vals*m - x0))[0] ** 2

#    airy_vals /= np.trapz(airy_vals, x_vals)  # Normalización
    airy_vals *= 1 #N_n  # Escalar para que coincida con la altura de prob
    # ax2.set_xlim(min(x_vals),10)
    # ax2.set_ylim(min(airy_vals), max(airy_vals))
    exact_solution_line.set_data(x_vals, airy_vals)

# Inicialización
oldE = energy(path)
counter = 1
norm = 0.0
maxx = 0.0
controlVar = 0
# Bucle principal

loopVar= True
while loopVar:
    element = random.randint(1, N - 1)  # Evitar los extremos (0 y N)
    change = (random.random() - 0.5) * 20.0 / 10.0 #De -5 a 5.
    if path[element] + change > 0.0:  # Evitar valores negativos
        path[element] += change  # Cambio temporal
        newE = energy(path)  # Nueva energía
        if newE > oldE and np.exp(-(newE - oldE)) <= random.random():
            path[element] -= change  # Rechazar cambio
        #path[element]= número flotante



        ele = path[element]


        if ele == xmax:
            prob[-1] += 1
            continue

        if ele > xmax or ele < xmin:
            controlVar += 1
            continue

        # Calcular la caja correspondiente en el histograma
        bin_index = int((ele - xmin) // anchoInt)
        prob[bin_index] += 1



        oldE = newE

    if counter % 1000 == 0:  # Actualizar la función de onda cada 100 pasos

        plt.pause(0.01)
        loading = counter*100/nsamples
        print(f"Avance: {loading} %")

        conversion = 1.0 / (nsamples * anchoInt)




        for i in range(nbins):
            pdfi = prob[i] * conversion
            pdf[i] = pdfi

        plotwf(pdf)
        plot_exact()  # Graficar la solución exacta

    counter += 1

    if (counter==(nsamples+1)):
        loopVar = False


print(f"Datos ignorados: {controlVar}")
SUM = np.sum(pdf)
print(f"Suma: {SUM * anchoInt}")  # Este valor debería ser aproximadamente 1.
