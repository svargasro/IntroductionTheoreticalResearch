import numpy as np
import matplotlib.pyplot as plt

# Parámetros (ajusta estos valores según tu problema)
mg = 1.0         # ejemplo: mg = 1.0
lambda1 = 0.1#1.0    # ejemplo: lambda1 = 1.0
kappa = 10      # ejemplo: kappa = 1.0

# Número de coeficientes a calcular (cuanto mayor, mejor aproximación en un rango razonable)
N_coef = 500

# Inicializamos la lista de coeficientes.
# NOTA: En Python, el índice 0 corresponde a a_0, etc.
a = [0]*N_coef

# Condiciones iniciales:
# Para la solución impar, elegimos a0 = 0 y a1 = 1
epsilon = 1e-5
a[0] = epsilon + epsilon*1j  # se fuerza como número complejo para incluir la unidad imaginaria i
a[1] = 1.0 + epsilon*1j

# La fórmula para a_2 es:
#   a_2 = -(i*lambda1 + kappa^2)*a_0 / 2
# Como a0 = 0, en este caso a2 = 0
if N_coef > 2:
    a[2] = - (1j*lambda1 + kappa**2) * a[0] / 2

# Usamos la recurrencia para n >= 1:
#   a_{n+2} = -{ mg*a_{n-1} + [i*lambda1*(2n+1) + kappa^2]*a_n } / [(n+2)(n+1)]
for n in range(1, N_coef - 2):
    numerator = mg * a[n-1] + (1j*lambda1*(2*n+1) + kappa**2)*a[n]
    denominator = (n+2) * (n+1)
    a[n+2] = - numerator / denominator

# (Opcional) Imprimir los coeficientes calculados
print("Coeficientes a_n:")
for n, coef in enumerate(a):
    print(f"a_{n} = {coef}")

# Definimos la función psi(x) a partir de la serie
def psi(x, coeffs):
    """
    Evalúa la función psi(x) = sum_{n=0}^{N_coef-1} a_n x^n
    """
    s = 0.0 + 0.0j
    for n, a_n in enumerate(coeffs):
        s += a_n * (x**n)
    return s

# Creamos un rango de valores para x.
# El rango debe elegirse de acuerdo a la convergencia de la serie.
x_vals = np.linspace(0, 4, 1000)  # ejemplo: de -1 a 1

# Evaluamos psi(x) en cada punto.
# Dado que psi(x) es compleja, graficamos la parte real (o la magnitud, según se desee)
psi_vals = np.array([psi(x, a) for x in x_vals])

amplitud = np.abs(psi_vals)**2
# Graficamos la parte real de psi(x)
plt.figure(figsize=(8, 5))
plt.plot(x_vals, psi_vals.real, label=r"Re[$\psi(x)$]")
plt.plot(x_vals, psi_vals.imag, label=r"Im[$\psi(x)$]", linestyle="--")
plt.plot(x_vals, amplitud, label=r"[$|\psi(x)|^2$]", linestyle="--")
plt.title(r"Serie de potencias: $\psi(x)=\sum_{n} a_nx^n$")
plt.xlabel("x")
plt.ylabel(r"$\psi(x)$")
plt.legend()
plt.grid(True)
plt.savefig("function.pdf")
