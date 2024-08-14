import numpy as np
from PC_SAFT import PCSAFT, flash
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Solve_ChemEQ import solve_ChemEQ_gekko
from PC_SAFT import BPT_gekko
import time

#%% System Test for Methane, Ethane, and Butane


# T = 233.15
# x = np.array([.09998, .3, .6])  # Mole fraction
# m = np.array([1, 1.6069, 2.0020])  # Number of segments
# σ = np.array([3.7039, 3.5206, 3.6184])  # Temperature-Independent segment diameter σ_i (Aᵒ)
# ϵ_k = np.array([150.03, 191.42, 208.11])  # Depth of pair potential / Boltzmann constant (K)
# k_ij = np.array([[0.00E+00, 3.00E-04, 1.15E-02],
#                  [3.00E-04, 0.00E+00, 5.10E-03],
#                  [1.15E-02, 5.10E-03, 0.00E+00]])
# κ_AB = np.array([0, 0, 0])
# ϵ_AB_k = np.array([0, 0, 0])
#
# # mix_l = PCSAFT(T, x, m, σ, ϵ_k, k_ij, phase='liquid', κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k, η=.402)
# # print(mix_l.da_dx())
# P_sys = 1000000
# yg = [.7, .2, .10]
# Pg = P_sys
# y, P = flash(x, yg, T, Pg, m, σ, ϵ_k, k_ij, 'Bubble_T')
# print(y, P)

#%% System test for CO2, MEA, H2O, with N2, and O2 in the vapor phase

MW_CO2 = 44.01/1000
MW_MEA = 61.08/1000
MW_H2O = 18.02/1000


def get_x(α, w_MEA):

    x_MEA = ((1 + α + (MW_MEA/MW_H2O))*(1-w_MEA)/w_MEA)**-1
    x_CO2 = x_MEA*α
    x_H2O = 1 - x_CO2 - x_MEA

    return [x_CO2, x_MEA, x_H2O]


def liquid_density(Tl, x, df_param):

    x_CO2, x_MEA, x_H2O = x

    MWs_l = np.array([44.01, 61.08, 18.02]) / 1000  # kg/mol

    MWT_l = sum([x[i] * MWs_l[i] for i in range(len(x))])

    a1, b1, c1 = [-5.35162e-7, -4.51417e-4, 1.19451]
    a2, b2, c2 = [-3.2484e-6, 0.00165, 0.793]

    V_MEA = MWs_l[1]*1000 / (a1 * Tl ** 2 + b1 * Tl + c1)  # mL/mol
    V_H2O = MWs_l[2]*1000 / (a2 * Tl ** 2 + b2 * Tl + c2)   # mL/mol

    # a, b, c, d, e = df_param['molar_volume'].values()
    a, b, c, d, e = 10.57920122, -2.020494157, 3.15067933, 192.0126008, -695.3848617

    V_CO2 = a + (b + c * x_MEA) * x_MEA * x_H2O + (d + e * x_MEA) * x_MEA * x_CO2

    V_l = V_CO2 * x_CO2 + x_MEA * V_MEA + x_H2O * V_H2O # Liquid Molar Volume (mL/mol)
    V_l = V_l*1e-6  # Liquid Molar Volume (mL/mol --> m3/mol)

    rho_mol_l = V_l**-1  # Liquid Molar Density (m3/mol --> mol/m3)
    rho_mass_l = rho_mol_l*MWT_l  # Liquid Mass Density (mol/m3 --> kg/m3)

    return rho_mol_l, rho_mass_l


df = pd.read_excel(r'C:\Users\Tanner\Documents\git\MEA\data\Jou_1995_30%.xlsx', sheet_name='T = 40')
P_CO2_data, α_data, P_data = df['P_CO2'].to_numpy()*1000, df['alpha'].to_numpy(), df['P'].to_numpy()*1000
P_interp = interp1d(α_data, P_data, kind='cubic')

α = .4
Tl = 313 # K
x = get_x(α, .3)
rho_mol_l, _ = liquid_density(Tl, x, [])
Cl = [x[i] * rho_mol_l for i in range(len(x))]
Cl_true = solve_ChemEQ_gekko(Cl, Tl)
x_true = Cl_true / (sum(Cl_true)).astype('float')
x = x_true[:3]


m = np.array([2.0729, 3.0353, 1.9599])  # Number of segments
σ = np.array([2.7852, 3.0435, 2.362])  # Temperature-Independent segment diameter σ_i (Aᵒ)
ϵ_k = np.array([169.21, 277.174, 279.42]) # Depth of pair potential / Boltzmann constant (K)
k_ij = np.array([[0.0, .16, .065],
                 [.16, 0.0, -.18],
                 [.065, -.18, 0.0]])
κ_AB = np.array([0, .037470, .2039])
ϵ_AB_k = np.array([0, 2586.3, 2059.28])

yg = [3e-04, 3e-07, .99]
Pg = 500000

start = time.time()



y, P = BPT_gekko(x, yg, Tl, Pg, m, σ, ϵ_k, k_ij, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)

end = time.time()
print(end - start)

start = time.time()

y, P = flash(x, yg, Tl, Pg, m, σ, ϵ_k, k_ij, flash_type='Bubble_T', κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)

end = time.time()
print(end - start)
