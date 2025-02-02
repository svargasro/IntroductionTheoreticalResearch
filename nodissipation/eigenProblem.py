import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ---------------------------
# Parámetros del problema
# ---------------------------
lambda1 = 1.0   # Parámetro lambda_1 (real)
m = 1.0         # Masa
g = 1.0         # Gravedad
a = 0.0         # Límite inferior del dominio
blim = 1     # Límite derecho del dominio (anteriormente "b")
N = 200         # Número de puntos de la discretización
dx = (blim - a) / (N - 1)
x = np.linspace(a, blim, N)

# ---------------------------
# Construcción de la matriz del operador (Hamiltoniano)
# ---------------------------
#
# La ecuación diferencial es:
#
#   psi''(x) + i*lambda1*(2*x*psi'(x) + psi) + m*g*x*psi = E*psi
#
# Se discretiza utilizando diferencias finitas:
#
#   psi''(x) ≈ (psi[n-1] - 2 psi[n] + psi[n+1]) / dx²
#   psi'(x)  ≈ (psi[n+1] - psi[n-1]) / (2*dx)
#
# Por lo que el término 2x*psi'(x) se aproxima como:
#
#   2x*psi'(x) ≈ x[n]*(psi[n+1] - psi[n-1]) / dx
#
# La ecuación en el punto n (interior) se puede escribir como:
#
# [1/dx² - i*lambda1*x[n]/dx] * psi[n-1]
# + [-2/dx² + i*lambda1 + m*g*x[n]] * psi[n]
# + [1/dx² + i*lambda1*x[n]/dx] * psi[n+1]
# = E * psi[n]
#

# Creamos la matriz en formato LIL para facilitar la asignación
A = sp.lil_matrix((N, N), dtype=complex)

# Rellenamos la matriz para los puntos interiores (n = 1, 2, ..., N-2)
for n in range(1, N-1):
    coef_left  = 1.0/dx**2 - 1j * lambda1 * x[n] / dx
    coef_diag  = -2.0/dx**2 + 1j * lambda1 #+ m * g * x[n]
    coef_right = 1.0/dx**2 + 1j * lambda1 * x[n] / dx

    A[n, n-1] = coef_left
    A[n, n]   = coef_diag
    A[n, n+1] = coef_right

# Impone condiciones de contorno Dirichlet:
# psi(a)=0 y psi(blim)=0
# Esto se consigue haciendo que en las filas correspondientes
# la única entrada sea 1 y el resto 0.
A[0, :] = 0
A[0, 0] = 1.0
A[N-1, :] = 0
A[N-1, N-1] = 1.0

# Convertir la matriz a formato CSR para la resolución de eigenvalores
A_csr = A.tocsr()

# ---------------------------
# Resolución del problema eigenvalor
# ---------------------------
#
# Se buscan, por ejemplo, 5 eigenvalores (energías) con menor parte real.
#
num_eigvals = 5
eigenvals, eigenvecs = spla.eigs(A_csr, k=num_eigvals, which='SR')

# Ordenamos los eigenvalores según su parte real (de menor a mayor)
idx = np.argsort(eigenvals.real)
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

print("Eigenvalores (energías) aproximados:")
for i, E in enumerate(eigenvals):
    print(f"E[{i}] = {E}")

# ---------------------------
# Seleccionar y visualizar la función de onda fundamental
# ---------------------------
#
# Se asume que el estado base es el asociado al eigenvalor con la menor parte real.
psi_ground = eigenvecs[:, 0]

# Normalización de la función de onda (usando la regla del trapecio)
norm = np.sqrt(np.trapezoid(np.abs(psi_ground)**2, x))
psi_ground = psi_ground / norm

plt.figure(figsize=(8, 5))
plt.plot(x, np.abs(psi_ground)**2, label=r'$|\psi_{\mathrm{ground}}(x)|^2$')
plt.xlabel("x")
plt.ylabel(r"$|\psi(x)|^2$")
plt.title("Amplitud de probabilidad del estado fundamental")
plt.legend()
plt.grid(True)
plt.savefig("eigenSol.pdf")
