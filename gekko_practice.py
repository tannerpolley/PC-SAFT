import numpy as np
from gekko import GEKKO

m = GEKKO(remote=False)

tf = 5 * 60 * 60
dt = int(tf / 300) + 1
#m.time = np.linspace(0,tf,dt)
# for testing
m.time = [0, 0.01, 0.02, 0.03]

# Number of nodes
n = 41

# Length of domain
Lx = 1
Ly = Lx  # square domain

# Define for material
k = 1
rho = 8000
Cp = 500
G_total = 1
em = 1
sigma = 1
T_surr = 298
Z = 1

x_div = np.linspace(0, Lx, n)
y_div = np.linspace(Ly, 0, n)

[X, Y] = np.meshgrid(x_div, y_div)

# step size
dx = x_div[1] - x_div[0]
dy = y_div[1] - y_div[0]

# Temp. initialization
T = m.Array(m.Var, (n, n), value=290)

# Equation set-up

# Middle segments
for i in range(1, n - 1):
    for j in range(1, n - 1):
        m.Equation(rho * Cp * T[i, j].dt() == (k * \
                                               ((T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / dx ** 2 \
                                                + (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dy ** 2))
                   + (G_total - em * sigma * (T[i, j] ** 4 - T_surr ** 4)) / Z)

# Boundary Conditions
m.Equations([T[0, i] == 310 for i in range(1, n - 1)])
m.Equations([T[-1, i] == 310 for i in range(1, n - 1)])
m.Equations([T[i, 0] == 315 for i in range(1, n - 1)])
m.Equations([T[i, -1] == 315 for i in range(1, n - 1)])
m.Equations([T[0, 0] == 312, T[n - 1, 0] == 312, \
             T[0, n - 1] == 312, T[n - 1, n - 1] == 312])

m.options.IMODE = 4
m.solve(disp=False)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
for i in range(0, 4):
    for j in range(0, 4):
        plt.subplot(4, 4, i * 4 + j + 1)
        plt.plot(m.time, T[i, j].value)
plt.savefig('heat.png', dpi=600)
plt.show()
