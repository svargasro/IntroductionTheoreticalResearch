# QMCbouncer.py: g.s. wavefunction via path integration

import numpy as np
import random
from vpython import canvas, curve, label, vec, color, rate, exp

# Parameters
N = 100
dt = 0.05
g = 2.0
h = 0.00
maxel = 0
path = np.zeros(N + 1)  # Initialize path array
prob = np.zeros(201)  # Initialize probability array

# Setup trajectory display
trajec = canvas(width=300, height=500, title="Spacetime Trajectory")
trplot = curve(color=color.magenta, radius=1)

# Axes for trajectories
def trjaxs():
    curve(pos=[vec(-97, -100, 0), vec(100, -100, 0)], color=color.cyan)
    curve(pos=[vec(-65, -100, 0), vec(-65, 100, 0)], color=color.cyan)
    label(pos=vec(-65, 110, 0), text='t', box=False)
    label(pos=vec(-85, -110, 0), text='0', box=False)
    label(pos=vec(60, -110, 0), text='x', box=False)

# Wavefunction display
wvgraph = canvas(width=500, height=300, title="GS Prob")
wvplot = curve(color=color.yellow, radius=1)

# Axes for wavefunction
def wvfaxs():
    curve(pos=[vec(-200, -155, 0), vec(800, -155, 0)], color=color.cyan)
    curve(pos=[vec(-200, -150, 0), vec(-200, 400, 0)], color=color.cyan)
    label(pos=vec(-70, 420, 0), text='Probability', box=False)
    label(pos=vec(600, -220, 0), text='x', box=False)
    label(pos=vec(-200, -220, 0), text='0', box=False)

# Plot trajectory and wavefunction axes
trjaxs()
wvfaxs()

# Energy calculation method using numpy
def energy(arr):
    kinetic = 0.5 * np.sum(((arr[1:] - arr[:-1]) / dt) ** 2)
    potential = g * np.sum((arr[:-1] + arr[1:]) / 2)
    return kinetic + potential


# Method to plot xy trajectory
def plotpath(path):
    x_vals = 20 * path - 65
    y_vals = 2 * np.arange(len(path)) - 100
    trplot.clear()  # Borra los puntos existentes
    for x, y in zip(x_vals, y_vals):
        trplot.append(vec(x, y, 0))  # Agrega cada punto



# Method to plot wavefunction
def plotwf(prob):
    x_vals = 20 * np.arange(len(prob)) - 200
    y_vals = 0.5 * prob - 150
    wvplot.clear()  # Borra los puntos existentes
    for x, y in zip(x_vals, y_vals):
        wvplot.append(vec(x, y, 0))  # Agrega cada punto



# Initialization
oldE = energy(path)
counter = 1
norm = 0.0
maxx = 0.0

# Infinite loop
while True:
    rate(100)

    element = random.randint(1, N - 1)  # Avoid ends (0 and N)
    change = (random.random() - 0.5) * 20.0 / 10.0
    if path[element] + change > 0.0:  # No negative paths
        path[element] += change  # Temporary change
        newE = energy(path)  # New trajectory E
        if newE > oldE and exp(-(newE - oldE)) <= random.random():
            path[element] -= change  # Reject change
            plotpath(path)
        ele = int(path[element] * 1250.0 / 100.0)
        maxel = max(maxel, ele)
        prob[ele] += 1
        oldE = newE

    if counter % 100 == 0:  # Plot wavefunction every 100 steps
        maxx = max(maxx, np.max(path))
        h = maxx / maxel if maxel > 0 else 0
        firstlast = h * 0.5 * (prob[0] + prob[maxel])
        norm = np.sum(prob) * h + firstlast
        plotwf(prob)
    counter += 1
