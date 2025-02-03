import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.special import airy, ai_zeros
from scipy.integrate import simpson
# ---------------------------
# Parámetros del problema
# ---------------------------
lambda1 = 0.3 #1.0   # Parámetro lambda_1 (real)
m = 1.0/2.0         # Masa
g = 2.0         # Gravedad
a = 0.0         # Límite inferior del dominio
blim = 10.0     # Límite derecho del dominio (anteriormente "b")
N = 1000       # Número de puntos de la discretización
dx = (blim - a) / (N - 1)
x = np.linspace(a, blim, N)

def normalize(arr, x_vals): # Normalization using Simpson's method
    integral = simpson(arr, x_vals)
    if integral != 0:
        arr /= integral
    return arr


# ---------------------------
# Construcción de la matriz del operador (Hamiltoniano)
# ---------------------------
#
# La ecuación diferencial es:
#
#   psi''(x) + i*lambda1*(2*x*psi'(x) + psi) + m*g*x*psi = E*psi  con E<0
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
    coef_diag  = -2.0/dx**2 + 1j * lambda1 + m * g * x[n]
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

#Condición sobre el borde se hace 0
# A[N-1, :] = 0
# A[N-1, N-1] = 1.0


# # ---------------------------
# # Condición de frontera en x = blim (derecha)
# # ---------------------------
# # En lugar de imponer psi(blim)=0, se desea dejar "libre" la frontera.
# # Una opción es usar una condición de Neumann:
# #       psi'(blim) = 0
# # Aproximamos la derivada en el último punto con diferencias finitas de primer orden:
# #       psi'(blim) ≈ (psi[N-1] - psi[N-2]) / dx = 0  =>  psi[N-1] = psi[N-2]

# # Para imponer esta condición, sustituimos la última fila de la matriz.
# A[N-1, :] = 0
# A[N-1, N-2] = -1.0
# A[N-1, N-1] = 1.0



# #--------------------------
# #Condición de frontera en blim libre
# #--------------------------
# # --- En x = blim (n = N-1) usamos el esquema unidireccional ---
# # Verificamos que N sea suficiente para utilizar este esquema (N >= 4)
# if N < 4:
#     raise ValueError("El número de puntos N debe ser al menos 4 para aplicar el esquema unidireccional en blim.")

# i = N - 1  # índice para x = blim
# xi = x[i]

# # Coeficientes para la segunda derivada en x = blim:
# coef_psi_i   =  2.0 / dx**2
# coef_psi_im1 = -5.0 / dx**2
# coef_psi_im2 =  4.0 / dx**2
# coef_psi_im3 = -1.0 / dx**2

# # Aproximación para la primera derivada en x = blim:
# # psi'(x_i) ≈ (-3 psi[i] + 4 psi[i-1] - psi[i-2])/(2dx)
# # Al multiplicar por 2*x_i, el coeficiente se distribuye:
# # para psi[i]:    -3*x_i/(dx)
# # para psi[i-1]:   4*x_i/(dx)
# # para psi[i-2]:   - x_i/(dx)
# #
# # Además, aparece el término i*lambda1*psi(x) (con psi[i]).
# # Y el potencial: m*g*x_i*psi[i].
# #
# # Así, para el nodo i se tienen:
# # psi[i]:
# coef_i = coef_psi_i \
#          - 1j*lambda1*(3*xi/dx) + 1j*lambda1 + m*g*xi
# # psi[i-1]:
# coef_im1 = coef_psi_im1 + 1j*lambda1*(4*xi/dx)
# # psi[i-2]:
# coef_im2 = coef_psi_im2 - 1j*lambda1*(xi/dx)
# # psi[i-3]:
# coef_im3 = coef_psi_im3

# # Asignamos estos coeficientes en la fila i:
# A[i, i]   = coef_i
# A[i, i-1] = coef_im1
# A[i, i-2] = coef_im2
# A[i, i-3] = coef_im3
# #----------------------------

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
numEigen = 0
psi_ground = eigenvecs[:, numEigen]

# Normalización de la función de onda (usando la regla del trapecio)
norm = np.sqrt(np.trapezoid(np.abs(psi_ground)**2, x))

psi_ground = psi_ground / norm

#psi_ground = normalize(psi_ground,x)
def psi_exact2(x, numEigen, lambda1, m, g, blim):
    # lambda1=0
    return (2/blim)*np.power(np.sin((numEigen+1)*np.pi*x/blim),2)*(1+((lambda1**2)*(x**4)/4.0))

def airy_exact2(x, m, g):
    # hbar=1
    # m_neutron=m
    # l_g_neutron = (hbar*2 / (2 * m_neutron**2 * g))*(1/3)
    z_n = -ai_zeros(10)[0] # Primeras 10 raíces negativas
    z_i = z_n[0]
    #psi_1_neutron = (airy(x / l_g_neutron - z_i)[0]/np.sqrt(abs(airy(-z_i)[1])))**2
    psi_1_neutron = (airy(x - z_i)[0])**2
    psi_1_neutron = normalize(psi_1_neutron,x)
    # norm = np.sqrt(np.trapezoid(psi_1_neutron, x))
    # psi_1_neutron = psi_1_neutron / norm

    return psi_1_neutron






psi_ground_exact = psi_exact2(x,numEigen,lambda1,m,g,blim)
airy_ground_exact = airy_exact2(x,m,g)

plt.figure(figsize=(8, 5))
plt.plot(x, np.abs(psi_ground)**2, label=r'$|\psi_{0}^{\mathrm{dissipated}}(x)|^2$')
#plt.plot(x, psi_ground_exact, label=r'$|\psi_{\mathrm{ground}}(x)|^2 exact$',linestyle="--")
plt.plot(x, airy_ground_exact, label=r'$|\psi_{0}^{\mathrm{not\text{ }dissipated}}(x)|^2$',linestyle="--")
plt.xlabel("Position x [arb. units]")
plt.ylabel(r"Probability Density $|\psi_0(x)|^2$")
plt.title(r"Probability Density of the Quantum Bouncer with dissipation. $\lambda_1=0.3$")
plt.legend()
plt.grid(True)
plt.savefig("eigenSol.pdf")
