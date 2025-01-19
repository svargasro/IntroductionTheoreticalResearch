import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy

# Define los parámetros
z_n = 2.338107410459767  # Primer cero de Ai(x), conocido
z_E = 0   # Puedes ajustar el desplazamiento si es necesario

# Cálculo de N_n
_, Ai_prime, _, _ = airy(-z_n)  # Obtén el valor de Ai'(-z_n)
N_n = np.sqrt(abs(Ai_prime))    # N_n = sqrt(|Ai'(-z_n)|)

# Define la función ψ(z)
def psi(z):
    Ai, _, _, _ = airy(z - z_n)  # Ai(z - z_E)
    return N_n * Ai

# Rango de z para la gráfica
z = np.linspace(-5, 5, 500)
psi_values = psi(z)**2

# Graficar
plt.figure(figsize=(8, 6))
plt.plot(z, psi_values, label=r"$\psi(z)$", color='blue')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.axvline(z_E, color='red', linestyle='--', label=f"$z_E = {z_E}$")
plt.title(r"Función $\psi(z)$", fontsize=14)
plt.xlabel(r"$z$", fontsize=12)
plt.ylabel(r"$\psi(z)$", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.savefig("output.pdf")
