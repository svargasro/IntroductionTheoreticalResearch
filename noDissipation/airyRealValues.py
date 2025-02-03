import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy, ai_zeros

#Plantilla subatómica
# Def de constantes
hbar = 1.0545718e-34 # Reduced Planck's constant (J·s)
g = 9.81 # [m/s^2]
m_neutron = 1#1.675e-28 #-27 # Masa del neutrón (kg)

# Longitud característica del neutrón
l_g_neutron = (hbar*2 / (2 * m_neutron**2 * g))*(1/3)

# Raíces de la función de Airy
z_n = -ai_zeros(10)[0] # Primeras 10 raíces negativas

# Energías cuantizadas
E_n_neutron = z_n * m_neutron * g * l_g_neutron

# Posibles valores de posición (Normalized)
x_neutron = np.linspace(0, 5 * l_g_neutron, 1000)

# Estado base de la función de onda
psi_1_neutron = airy(x_neutron / l_g_neutron - z_n[0])[0] / np.sqrt(abs(airy(-z_n[0])[1]))

# Gráfica del estado base de la función de onda
plt.figure(figsize=(10, 6))
plt.plot(x_neutron, psi_1_neutron, label=f'Ground state ($n=1$, $E_1={E_n_neutron[0]:.2e}$ J)')
plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.title("Ground State Wavefunction of the Quantum Bouncer (Neutron)", fontsize=14)
plt.xlabel("Position $x$ (m)", fontsize=12)
plt.ylabel("Wavefunction $\\psi(x)$", fontsize=12)
plt.legend(fontsize=10)
plt.grid()
plt.savefig("Nueva.pdf")
#plt.show()

# # Energía del estado base
# E_1_neutron = z_n[0] * m_neutron * g * l_g_neutron # Usamos z_n[0]

# # Gráfica de la energía del estado base para el neutrón
# plt.figure(figsize=(10, 6))
# plt.hlines(E_1_neutron, 0, 1, label=f'Ground level ($n=1$, $E_1={E_1_neutron:.2e}$ J)', color='C0')
# plt.title("Ground State Energy of the Quantum Bouncer (Neutron)", fontsize=14)
# plt.xlabel("Quantum Level", fontsize=12)
# plt.ylabel("Energy $E_1$ (J)", fontsize=12)
# plt.yticks([E_1_neutron], labels=[f"$E_1$"])
# plt.xticks([])
# plt.grid(axis='y', linestyle='--', linewidth=0.5)
# plt.legend(fontsize=10)
# plt.show()
