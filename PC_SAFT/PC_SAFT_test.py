import numpy as np
from PC_SAFT import PCSAFT, flash

#%% System Test for Methane, Ethane, and Butane


T = 233.15
x = np.array([.09998, .3, .6])  # Mole fraction
m = np.array([1, 1.6069, 2.0020])  # Number of segments
σ = np.array([3.7039, 3.5206, 3.6184])  # Temperature-Independent segment diameter σ_i (Aᵒ)
ϵ_k = np.array([150.03, 191.42, 208.11])  # Depth of pair potential / Boltzmann constant (K)
k_ij = np.array([[0.00E+00, 3.00E-04, 1.15E-02],
                 [3.00E-04, 0.00E+00, 5.10E-03],
                 [1.15E-02, 5.10E-03, 0.00E+00]])
κ_AB = np.array([0, 0, 0])
ϵ_AB_k = np.array([0, 0, 0])

# mix_l = PCSAFT(T, x, m, σ, ϵ_k, k_ij, phase='liquid', κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k, η=.402)
# print(mix_l.da_dx())
P_sys = 1000000
yg = [.7, .2, .10]
Pg = P_sys
y, P = flash(x, yg, T, Pg, m, σ, ϵ_k, k_ij, 'Bubble_T')
print(y, P)

#%% System test for CO2, MEA, H2O, with N2, and O2 in the vapor phase

# MW_CO2 = 44.01/1000
# MW_MEA = 61.08/1000
# MW_H2O = 18.02/1000
#
#
#
# def get_x(α, w_MEA):
#
#     x_MEA = ((1 + α + (MW_MEA/MW_H2O))*(1-w_MEA)/w_MEA)**-1
#     x_CO2 = x_MEA*α
#     x_H2O = 1 - x_CO2 - x_MEA
#
#     return [x_CO2, x_MEA, x_H2O]
#
#
# T = 393 # K
# w_MEA = .30
# α = .3004
# x = get_x(α, w_MEA)
# print(x)
# x = [.04, .1, .86]
# y = [.1, .013, .818]
#
# # 2B Association Scheme
# x = [.04, .1, .86]
# m = np.array([2.0729, 3.0353, 1.9599])  # Number of segments
# σ = np.array([2.7852, 3.0435, 2.362])  # Temperature-Independent segment diameter σ_i (Aᵒ)
# ϵ_k = np.array([169.21, 277.174, 279.42]) # Depth of pair potential / Boltzmann constant (K)
# k_ij = np.array([[0.0, .16, .065],
#                  [.16, 0.0, -.18],
#                  [.065, -.18, 0.0]])
# κ_AB = np.array([0, .037470, .2039])
# ϵ_AB_k = np.array([0, 2586.3, 2059.28])
#
# P_sys = 109180
# #
# mix_l = PCSAFT(T, x, m, σ, ϵ_k, k_ij, phase='liquid', P_sys=P_sys, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
# print(mix_l.ρ())

# # Vapor
# y = [.1, .013, .818, .069]
# m = np.array([2.0729, 1.9599, 1.2053, 1.1335])  # Number of segments
# σ = np.array([2.7852, 2.362, 3.3130, 3.1947])  # Temperature-Independent segment diameter σ_i (Aᵒ)
# ϵ_k = np.array([169.21, 279.42, 90.96, 114.43]) # Depth of pair potential / Boltzmann constant (K)
# k_ij = np.array([[0.0,   .065, -.0149, -.04838],
#                  [.065,   0.0,    0.0, 0.0],
#                  [-.0149, 0.0,    0.0, 0.0],
#                  [-.04838, 0.0, 0.0, -.00978]])
#
# κ_AB = np.array([0, 0, 0, 0])
# ϵ_AB_k = np.array([0, 0, 0, 0])
#
# mix_v = PCSAFT(T, y, m, σ, ϵ_k, k_ij, phase='vapor', P_sys=P_sys, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
# print(mix_v.φ())