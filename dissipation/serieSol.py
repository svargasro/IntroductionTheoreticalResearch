import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Definir constantes
a1 = 1
m = (1.0/2)
g = 2
k = np.pi
lambd = 0.04

# Definir la función psi(x)
def psi(x,lambd):
    term1 = (1 + 1j * (lambd * x**2) / 2) * np.sin(k * x)

    term2 = m * g * ( (1/4) * k**4 * x**2 * ((2 - (k * x)**2) * np.cos(k*x) - 4 * k * x * np.sin(k * x))
                      + (1/4) * (k * x)**3 * np.sin(k * x)
                      - (1/2) * (k * x)**2 * np.cos(k * x) )

    term3 = (1j * lambd * x / 8) * ( x * ( k**6 * (12 - x) * x**2 + 6 * k**3 * (3 * k**2 - 2 * k + 1) ) * np.sin(k * x)
                                    + k**4 * x**2 * ( k**4 * x**2 - 36 * k**2 + 9 * k + 1 ) * np.cos(k * x)
                                    - 6 * k**2 * (k + 1) * np.cos(k * x) )

    return a1 * (term1 + term2 + term3)



def normalize(arr, x_vals): # Normalization using Simpson's method
    integral = simpson(arr, x_vals)
    if integral != 0:
        arr /= integral
    return arr

# Valores de x
x = np.linspace(0, 1, 1000)

# Evaluar la función
psi_vals = psi(x,lambd)
psi_vals_lamb0 = psi(x,0)


#psi_vals = normalize(psi_vals,x)
amplitud = np.abs(psi_vals)**2

amplitud_lamb0 = np.abs(psi_vals_lamb0)**2

# Graficar la parte real e imaginaria
plt.figure(figsize=(10, 5))
#plt.plot(x, psi_vals.real, label="Re(ψ(x))", color="blue")
#plt.plot(x, psi_vals.imag, label="Im(ψ(x))", color="red")
plt.plot(x, amplitud, label=r"$|\psi(x)|^2$")
plt.plot(x, amplitud_lamb0, label=r"$|\psi(x)|^2$ $\lambda=0$", linestyle="dashed")

plt.axhline(0, color='black', linewidth=0.5, linestyle="dotted")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.ylim(0,2)
plt.legend()
plt.title("Función ψ(x) con constantes = 1")
plt.grid()
plt.savefig("output.pdf")
