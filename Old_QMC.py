# QMC.py: Quantum Monte Carlo, Feynman path integration

import random
from visual.graph import *

N = 100
M = 101
xscale = 10

path = zeros([M], float)
prob = zeros([M], float)  # Initialize

trajec = display(width=300, height=500, title="Spacetime Trajectories")
trplot = curve(y=range(0, 100), color=color.magenta, display=trajec)

def trjaxs():  # axis
    trax = curve(pos=[(-97, -100), (100, -100)], color=color.cyan, display=trajec)
    label(pos=(0, -110), text="0", box=0, display=trajec)
    label(pos=(60, -110), text="x", box=0, display=trajec)

wvgraph = display(x=340, y=150, width=500, height=300, title="Ground State")
wvplot = curve(x=range(0, 100), display=wvgraph)
wvfax = curve(color=color.cyan)

def wvfaxs():  # axis for probability
    wvfax = curve(pos=[(-600, -155), (800, -155)], display=wvgraph, color=color.cyan)
    curve(pos=[(0, -150), (0, 400)], display=wvgraph, color=color.cyan)
    label(pos=(-80, 450), text="Probability", box=0, display=wvgraph)
    label(pos=(600, -220), text="0", box=0, display=wvgraph)
    label(pos=(0, -220), text="0", box=0, display=wvgraph)

trjaxs()
wvfaxs()  # plot axes

def energy(path):  # HO energy
    sums = 0
    for i in range(0, N - 2):
        sums += (path[i + 1] - path[i]) * (path[i + 1] - path[i])
        sums += path[i + 1] * path[i + 1]
    return sums

def plotpath(path):  # plot trajectory in xy scale
    for j in range(0, N):
        trplot.x[j] = 20 * path[j]
        trplot.y[j] = 2 * j - 100

def plotwf(prob):  # plot probability
    for i in range(0, 100):
        wvplot.color = color.yellow
        wvplot.x[i] = 8 * i - 400  # convenient coordinates
        wvplot.y[i] = 4.0 * prob[i] - 150  # for centered figure

oldE = energy(path)  # find E of path

while True:
    rate(10)  # slows the painting
    element = int(N + random.random())  # pick random element
    change = 2.0 * (random.random() - 0.5)  # Metropolis algorithm
    path[element] += change  # Change path
    newE = energy(path)  # Find new E
    if newE > oldE and math.exp(-newE + oldE) <= random.random():
        path[element] -= change  # Reject
        plotpath(path)  # plot resulting last trajectory

    elem = int(path[element] * 16 + 50)  # if path = 0, elem = 50
    if elem < 0:
        elem = 0  # negative case not allowed
    if elem > 100:
        elem = 100  # if exceed max
    prob[elem] += 1  # increase probability for that value
    plotwf(prob)  # plot probability
    oldE = newE
