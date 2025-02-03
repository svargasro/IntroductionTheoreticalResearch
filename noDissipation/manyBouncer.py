import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.integrate import simpson
import os

# Parameters
N = 1000  # Path length
dt = 0.0001
g = 2.0  # gravity
m = 1.0  # mass
nbins = 1000  # number of bins in the histogram
xmin, xmax = 0, 80
nsamples = 100000 + 10000  # number of random samples
ancho_int = (xmax - xmin) / nbins  # Width of histogram bins
x0 = 2.338107410459767  # First zero of airy function

# File to save/load data
data_file = "probability_data.txt"

# Plot initialization
fig, ax2 = plt.subplots(figsize=(8, 5))
ax2.set_title("Probability Density of the Quantum Bouncer without Dissipation")
ax2.set_xlabel("Position x [arb. units]")
ax2.set_ylabel(r"$|\psi_0(x)|^2$")
ax2.grid(True)
wavefunction_line, = ax2.plot([], [], linewidth=1, label="Simulated Solution")
exact_solution_line, = ax2.plot([], [], linestyle='--', label="Exact Solution")
ax2.legend()

def energy(arr):  # Energy function
    kinetic = 0.5 * m * np.sum(((arr[2:] - arr[:-2]) / (2 * dt)) ** 2)  # Kinetic energy
    potential = g * np.sum((arr[:-1] + arr[1:]) / 2)  # Average potential energy
    return kinetic + potential

def normalize(arr, x_vals):  # Normalization using Simpson's method
    integral = simpson(arr, x_vals)
    if integral != 0:
        arr /= integral
    return arr

def plotwf(prob):  # Probability density function plot
    x_vals = np.linspace(xmin, xmax, nbins)
    wavefunction_line.set_data(x_vals, prob)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 0.6)
    fig.savefig("manyProbPlot.pdf")

def plot_exact():  # Exact solution plot
    x_vals = np.linspace(0, 10, 5000)
    airy_vals = airy((x_vals - x0))[0] ** 2
    airy_vals = normalize(airy_vals, x_vals)
    exact_solution_line.set_data(x_vals, airy_vals)

if os.path.exists(data_file):
    print("Data file exists. Loading data...")
    average_prob = np.loadtxt(data_file)
    plot_exact()
    plotwf(average_prob)
else:
    print("Data file not found. Starting calculations...")

    # Simulation repeats
    num_repeats = 50
    all_probs = np.zeros((num_repeats, nbins))

    for repeat in range(num_repeats):
        print(f"Starting simulation {repeat + 1} of {num_repeats}")

        # Arrays initialization
        path = np.full(N + 1, 0.5)  # Path initialization
        prob = np.zeros(nbins)  # Histogram initialization
        oldE = energy(path)  # Initial energy

        for counter in range(int(nsamples)):  # Main loop
            element = random.randint(1, N - 1)  # Avoid the extremes
            change = (random.random() - 0.5) * 2.0  # Random value between -1 and 1

            if path[element] + change > 0.0:  # Evaluate change only if new value is positive
                path[element] += change
                newE = energy(path)

                if newE > oldE and np.exp(-(newE - oldE)) <= random.random():
                    path[element] -= change  # Reject the change

                ele = path[element]

                if ele >= xmax:
                    prob[-1] += 1
                elif xmin <= ele < xmax:
                    bin_index = int((ele - xmin) // ancho_int)
                    prob[bin_index] += 1  # Update histogram

                oldE = newE

            if counter % 1000 == 0:
                loading = (counter * 100) / nsamples
                print(f"Simulation {repeat + 1}: Progress {loading:.2f}%")

        # Normalize probability density function
        conversion_factor = 1.0 / (nsamples * ancho_int)
        prob *= conversion_factor

        all_probs[repeat] = prob  # Store current result

    # Calculate the average result
    average_prob = np.mean(all_probs, axis=0)

    # Save data to file
    np.savetxt(data_file, average_prob)
    print(f"Data saved to {data_file}")

    # Plot results
    plot_exact()
    plotwf(average_prob)

