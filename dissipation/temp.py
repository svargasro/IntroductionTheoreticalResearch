import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import simpson

def psi(x, N=30, a1=1, mg=1, lambda1=1, k=1):
    """
    Evalúa la función Psi(x) definida por la serie:

      Psi(x) = a1 * x * [ S1(x) + mg * S2(x) ]

    donde:

      S1(x) = sum_{n=0}^{N-1} (-1)^n/(2n+1)! * [ (2n+1)*i*lambda1 + k^2 ]^n * x^(2n)

      S2(x) = sum_{n=0}^{N-1} (-1)^n/(2n)! * n(n+1) * [ (2n+3)*i*lambda1 + k^2 ]^n * x^(2n+1)

    Los parámetros a1, mg, lambda1 y k son ajustables.
    """
    s1 = 0 + 0j
    s2 = 0 + 0j
    for n in range(N):
        # Primer sumatorio:
        denom1 = math.factorial(2 * n + 1)
        term_factor1 = (((2 * n + 1) * 1j * lambda1) + k**2) ** n
        term1 = ((-1)**n / denom1) * term_factor1 * (x ** (2 * n))
        s1 += term1

        # Segundo sumatorio:
        denom2 = math.factorial(2 * n)
        term_factor2 = (((2 * n + 3) * 1j * lambda1) + k**2) ** n
        term2 = ((-1)**n / denom2) * (n * (n + 1)) * term_factor2 * (x ** (2 * n + 1))
        s2 += term2

    return a1 * x * (s1 + mg * s2)

# =============================
# Parámetros ajustables:
a1      = 1      # Factor a1
mg      = 1      # mg
lambda1 = 0      # lambda
k       = np.pi/4 #np.pi # k

N_terms = 60     # Número de términos para la serie (puedes aumentarlo o reducirlo según convenga)
# =============================

# Rango de x en el que se evaluará la función
x_vals = np.linspace(0, 2, 1000)

# Se evalúa Psi(x) para cada x del rango (la función es compleja)
psi_vals = np.array([psi(x, N=N_terms, a1=a1, mg=mg, lambda1=lambda1, k=k) for x in x_vals])

def normalize(arr, x_vals): # Normalization using Simpson's method
    integral = simpson(arr, x_vals)
    if integral != 0:
        arr /= integral
    return arr

psi_vals = normalize(psi_vals,x_vals)



# Extraer partes real e imaginaria para graficar
psi_real = psi_vals.real
psi_imag = psi_vals.imag

# Graficar
plt.figure(figsize=(10, 6))
#plt.plot(x_vals, psi_real, label=r"Re[$\Psi(x)$]", color='blue')
#plt.plot(x_vals, psi_imag, label=r"Im[$\Psi(x)$]", color='red')
plt.plot(x_vals, np.abs(psi_vals)**2, label=r"[$|\Psi(x)|^2$]", linestyle="--")
plt.ylim(0,1)
plt.xlabel("x")
plt.ylabel(r"$\Psi(x)$")
plt.title(r"Gráfica de $\Psi(x)$")
plt.legend()
plt.grid(True)
plt.savefig("Func.pdf")
