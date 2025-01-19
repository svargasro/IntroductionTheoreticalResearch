#!/usr/bin/env python3
# QMC.py: Quantum Monte Carlo, Feynman path integration

import random
import math
from vpython import canvas, curve, vector, label, rate, color

# Configuración inicial
N = 100
M = 101
xscale = 10

# Inicializar arreglos
path = [0.0] * M
prob = [0.0] * M

# Crear gráficos y configuraciones
trajec = canvas(title="Spacetime Trajectories", width=600, height=500)
trplot = curve(color=color.magenta)  # Trajectories plot

wvgraph = canvas(title="Ground State", width=600, height=400)
wvplot = curve(color=color.yellow)  # Probability plot

# Ejes de las gráficas
def draw_trajec_axes():
    curve(pos=[vector(-100, -100, 0), vector(100, -100, 0)], color=color.cyan, canvas=trajec)
    label(pos=vector(0, -110, 0), text="0", box=False, canvas=trajec)
    label(pos=vector(60, -110, 0), text="x", box=False, canvas=trajec)

def draw_prob_axes():
    curve(pos=[vector(-600, -155, 0), vector(800, -155, 0)], color=color.cyan, canvas=wvgraph)
    curve(pos=[vector(0, -150, 0), vector(0, 400, 0)], color=color.cyan, canvas=wvgraph)
    label(pos=vector(-80, 450, 0), text="Probability", box=False, canvas=wvgraph)
    label(pos=vector(600, -220, 0), text="0", box=False, canvas=wvgraph)
    label(pos=vector(0, -220, 0), text="0", box=False, canvas=wvgraph)

draw_trajec_axes()
draw_prob_axes()

# Función para calcular energía
def energy(path):
    sums = 0
    for i in range(0, N - 2):
        sums += (path[i + 1] - path[i])**2
        sums += path[i+1]**2
    return sums

# Graficar trayectorias
def plotpath(path):
    trplot.clear()
    for j in range(N):
        trplot.append(pos=vector(20 * path[j], 2 * j - 100, 0))

# Graficar función de onda
def plotwf(prob):
    wvplot.clear()
    for i in range(100):
        x_coord = 8 * i - 400
        y_coord = 4.0 * prob[i] - 150
        wvplot.append(pos=vector(x_coord, y_coord, 0))

# Inicializar energía
oldE = energy(path)

# Algoritmo principal
while True:
    rate(10)  # Controla la velocidad de la simulación
    element = int(N * random.random())  # Escoger un elemento aleatorio
    change = 2.0 * (random.random() - 0.5)  # Algoritmo Metropolis
    path[element] += change  # Cambiar el camino
    newE = energy(path)  # Calcular nueva energía

    if newE > oldE and math.exp(-newE + oldE) <= random.random():
        path[element] -= change  # Rechazar cambio si no cumple criterio

    plotpath(path)  # Graficar trayectoria resultante

    elem = int(path[element] * 16 + 50)  # Escalar a probabilidad
    elem = max(0, min(100, elem))  # Limitar rango
    prob[elem] += 1  # Incrementar probabilidad para ese valor
    plotwf(prob)  # Graficar probabilidad
    oldE = newE
