import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.integrate import simpson

#Fixed seed.
# SEED_VALUE = #200
# random.seed(SEED_VALUE)
# np.random.seed(SEED_VALUE)

# Parameters
N = 1000 #Path length
dt = 0.0001
g = 2.0 #gravity
m = 1.0 #mass
nbins = 1000 #number of bins in the histogram
xmin, xmax = 0, 80
nsamples = 100000+10000 #number of random samples
ancho_int = (xmax - xmin) / nbins #Width of histogram bins

#Arrays initialization
path =  np.full(N+1,0.5) #Path initialization
prob = np.zeros(nbins) #Histogram


x0 = 2.338107410459767 # First zero of airy function

# Plot inizialization
fig, ax2 = plt.subplots(figsize=(8,5))
ax2.set_title("Probability Density of the Quantum Bouncer without Dissipation")
ax2.set_xlabel("Position x [arb. units]")
ax2.set_ylabel(r"$|\psi_0(x)|^2$")
ax2.grid(True)
wavefunction_line, = ax2.plot([], [], linewidth=1, label="Simulated Solution")
exact_solution_line, = ax2.plot([], [], linestyle='--', label="Exact Solution")
ax2.legend()


def energy(arr): #Energy function
    kinetic = 0.5 *m* np.sum(((arr[2:] - arr[:-2]) / (2 * dt)) ** 2)  #Kinetic energy with central derivative
    potential = g * np.sum((arr[:-1] + arr[1:]) / 2)  # Average potential energy
    return kinetic + potential


def plotwf(prob): #probability density function plot
    x_vals = np.linspace(xmin, xmax, nbins)
    wavefunction_line.set_data(x_vals, prob)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 0.6)
    fig.savefig("probPlot.pdf")

def plot_exact(): #exact solution plot
    x_vals = np.linspace(0, 10, 5000)
    airy_vals = airy((x_vals - x0))[0] ** 2
    airy_vals= normalize(airy_vals,x_vals)
    exact_solution_line.set_data(x_vals, airy_vals)




def normalize(arr, x_vals): # Normalization using Simpson's method
    integral = simpson(arr, x_vals)
    if integral != 0:
        arr /= integral
    return arr


oldE = energy(path) #energy inizialization
control_var = 0
counter = 1

# Bucle principal
for _ in range(int(nsamples)): #Iterations
    element = random.randint(1, N - 1)  # Avoid the extremes
    change = (random.random() - 0.5) * 2.0  #Random value between -1 and 1

    
    if path[element] + change > 0.0: #The change is evaluated only and only if the new path[element] is higher than 0
        path[element] += change
        newE = energy(path)

        if newE > oldE and np.exp(-(newE - oldE)) <= random.random(): #If both conditions are met, the change is rejected.
            path[element] -= change

        ele = path[element]

        if ele >= xmax:
            prob[-1] += 1
        elif xmin <= ele < xmax:
            bin_index = int((ele - xmin) // ancho_int)
            prob[bin_index] += 1 #The histogram is updated
        else:
            control_var += 1

        oldE = newE

    if counter % (nsamples - 1) == 0:
        conversion_factor = 1.0 / (nsamples * ancho_int) #Histogram normalization: probability density function
        pdf = prob * conversion_factor
        plot_exact()
        plotwf(pdf)

    if counter % 1000 == 0:
        loading = (counter * 100) / nsamples
        print(f"Avance: {loading:.2f}%") #Progress percentage

    counter += 1

# print(f"Datos ignorados: {control_var}")
sum_pdf = np.sum(prob * conversion_factor)
print(f"Suma: {sum_pdf * ancho_int:.6f}")  # Este valor debería ser aproximadamente 1.
