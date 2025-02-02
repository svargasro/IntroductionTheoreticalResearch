import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros de la ecuación
lambda_1 = 1.0  # Puedes ajustar este valor
k = 1.0         # Puedes ajustar este valor

# Definimos la ecuación diferencial
# Separaremos psi en parte real (u[0]) e imaginaria (u[1])
def eq_diff(x, u):
    psi_real, psi_imag = u[0], u[1]
    dpsi_real_dx = u[2]
    dpsi_imag_dx = u[3]

    d2psi_real_dx2 = -lambda_1 * (2 * x * dpsi_imag_dx + psi_imag) - k**2 * psi_real
    d2psi_imag_dx2 = lambda_1 * (2 * x * dpsi_real_dx + psi_real) - k**2 * psi_imag

    return [dpsi_real_dx, dpsi_imag_dx, d2psi_real_dx2, d2psi_imag_dx2]

# Condiciones iniciales
x_min, x_max = -10, 10
psi_real_0 = 0.0  # Parte real inicial de psi en x = 0
psi_imag_0 = 0.0  # Parte imaginaria inicial de psi en x = 0
dpsi_real_dx_0 = 0.0  # Derivada inicial de la parte real de psi en x = 0
dpsi_imag_dx_0 = 0.0  # Derivada inicial de la parte imaginaria de psi en x = 0

# Resolución numérica
x_eval = np.linspace(x_min, x_max, 1000)  # Puntos donde se evaluará la solución
sol = solve_ivp(eq_diff, [x_min, x_max], [psi_real_0, psi_imag_0, dpsi_real_dx_0, dpsi_imag_dx_0], t_eval=x_eval)

# Extraemos las soluciones
x = sol.t
psi_real = sol.y[0]
psi_imag = sol.y[1]

# Gráfica de la solución
plt.figure(figsize=(10, 6))
plt.plot(x, psi_real*psi_imag, label="Re(psi)")
#plt.plot(x, psi_imag, label="Im(psi)")
plt.title("Solución de la ecuación diferencial")
plt.xlabel("x")
plt.ylabel("psi(x)")
plt.legend()
plt.grid()
plt.show()
