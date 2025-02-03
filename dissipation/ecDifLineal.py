import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Parámetros
lambda1 = 1.0  # Coeficiente lambda_1
m = 1.0        # Masa
g = 1.0        # Gravedad
a = 0.0        # Límite inferior del dominio
blim = 1         # Límite superior del dominio
k = (3*np.pi/blim)        # Coeficiente k (Autoenergía)
N = 100        # Número de puntos en la discretización
dx = (blim - a) / (N - 1)  # Espaciado en x
x = np.linspace(a, blim, N)  # Dominio espacial

# Matriz A (en formato lista de listas, para facilitar la construcción)
A = sp.lil_matrix((N, N), dtype=complex)

# Construcción de la matriz A
for n in range(1, N-1):  # Usamos n en lugar de j para evitar confusiones
    A[n, n-1] = 1 / dx**2 - 1j * lambda1 * x[n] / (2 * dx)  # Término para psi_{n-1}
    A[n, n] = -2 / dx**2 + 1j * lambda1 + k**2 + m * g * x[n]  # Término para psi_n
    A[n, n+1] = 1 / dx**2 + 1j * lambda1 * x[n] / (2 * dx)  # Término para psi_{n+1}

# Condición de contorno ajustada en x = 0
epsilon_a = 1e-5  # Valor cercano a cero para x = 0
A[0, 0] = 1  # Fija psi(0) = epsilon_a
b = np.zeros(N, dtype=complex)
b[0] = epsilon_a  # Ajusta el vector b para reflejar la condición

# Condición de contorno ajustada en x = b
epsilon_b = 1e-5  # Valor cercano a cero para x = b
n = N - 1  # Índice del último punto
A[n, n] = 1  # Fija psi(b) = epsilon_b
b[n] = epsilon_b



# Resolver el sistema lineal A psi = b
psi = spla.spsolve(A.tocsr(), b)

# Calcular la amplitud de probabilidad
amplitud = np.abs(psi)**2

# Graficar la amplitud de probabilidad
plt.plot(x, amplitud, label=r"$|\psi(x)|^2$")
plt.legend()
plt.xlabel("x")
plt.ylabel(r"$|\psi(x)|^2$")
plt.title("Amplitud de probabilidad de la función de onda")
plt.savefig("sol.pdf")
