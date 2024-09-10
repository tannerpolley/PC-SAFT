import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Get_True_Mol_Frac import get_true_mol_frac
from PC_SAFT_GEKKO import PC_SAFT_bubble_T


#%%
plt.figure(figsize=(10, 10))
T = 40
df = pd.read_csv(r'C:\Users\Tanner\Documents\git\MEA\data\data_sets_to_load\Jou_1995_VLE.csv')
df = df[df['temperature'] == T]
P_CO2_data, α_data, P_data = df['CO2_pressure'].to_numpy() * 1000, df['CO2_loading'].to_numpy(), df[
    'total_pressure'].to_numpy() * 1000
P_interp = interp1d(α_data, P_data, kind='cubic')

w_MEA = .3
Tl = 273.15 + T
P_CO2_g = []
α_range = np.linspace(α_data[0], α_data[-1], 21)
alpha = .0888
# for alpha in α_range:

x = get_true_mol_frac(alpha, w_MEA, Tl)
Pg = float(P_interp(alpha))
yg = [.01, .00001, .98]

m_seg = np.array([2.0729, 3.0353, 1.9599])  # Number of segments
σ = np.array([2.7852, 3.0435, 2.362])  # Temperature-Independent segment diameter σ_i (Aᵒ)
ϵ_k = np.array([169.21, 277.174, 279.42])  # Depth of pair potential / Boltzmann constant (K)
k_ij = np.array([[0.0, .16, .065],
                 [.16, 0.0, -.18],
                 [.065, -.18, 0.0]])
# κ_AB = np.array([0, .037470, .2039])
# ϵ_AB_k = np.array([0, 2586.3, 2059.28])


P_CO2_g.append(PC_SAFT_bubble_T(x, yg, Tl, Pg, m_seg, σ, ϵ_k, k_ij)/1e3)


print(P_CO2_g)
#%%
# plt.plot(α_data, P_CO2_data / 1e3, 'x')
# plt.plot(α_range, P_CO2_g, '--', label=f'P_CO2 - T = {T}')
# plt.xlabel("CO$_{2}$ Loading, mol CO$_{2}$/mol MEA", fontsize=16)
# plt.ylabel("CO$_{2}$ pressure, kPa", fontsize=16)
# plt.tick_params(labelsize=14)
# plt.tight_layout()
# plt.yscale('log')
# plt.legend()
# plt.show()


