from cProfile import label

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Get_True_Mol_Frac import get_true_mol_frac
from PC_SAFT import flash

#%%
plt.figure(figsize=(10, 10))
colors = ['red', 'blue', 'green', 'yellow', 'cyan']
etal = []
etav = []
P_range = []
T_range = []
alpha_range = []
a_assoc_l_list = []
a_assoc_v_list = []
a_res_l_list = []
for i, T in enumerate([40, 80]):
    df = pd.read_csv(r'data\data_sets_to_load\Jou_1995_VLE.csv')
    df = df[df['temperature'] == T]
    P_CO2_data, α_data, P_data = df['CO2_pressure'].to_numpy() * 1000, df['CO2_loading'].to_numpy(), df[
        'total_pressure'].to_numpy() * 1000
    P_interp = interp1d(α_data, P_data, kind='cubic')

    w_MEA = .3
    Tl = 273.15 + T
    P_CO2_list = []
    P_CO2_2_list = []
    α_range = np.linspace(α_data[0], α_data[-1], 11)
    for alpha in α_range:

        x = get_true_mol_frac(alpha, w_MEA, Tl)[:3]
        Pg = float(P_interp(alpha))
        yg = [.01, .00001, .98]

        m_seg = np.array([2.0729, 3.0353, 1.9599])  # Number of segments
        σ = np.array([2.7852, 3.0435, 2.362])  # Temperature-Independent segment diameter σ_i (Aᵒ)
        ϵ_k = np.array([169.21, 277.174, 279.42])  # Depth of pair potential / Boltzmann constant (K)
        k_ij = np.array([[0.0, .16, .065],
                         [.16, 0.0, -.18],
                         [.065, -.18, 0.0]])
        κ_AB = np.array([0, .037470, .2039])
        ϵ_AB_k = np.array([0, 2586.3, 2059.28])

        try:
            fug_l_CO2, P_CO2, ηl, ηv, a_assoc_l, a_assoc_v, a_res_l = flash(x, yg, Tl, Pg, m_seg, σ, ϵ_k, k_ij, flash_type='Bubble_T',
                                                                               κ_AB=κ_AB,
                                                                               ϵ_AB_k=ϵ_AB_k
                                                                               )
            fug_l_CO2_2, P_CO2_2, ηl_2, ηv_2, a_assoc_l_2, a_assoc_v_2, a_res_l_2 = flash(x, yg, Tl, Pg, m_seg, σ, ϵ_k, k_ij, flash_type='Bubble_T')
            P_CO2_list.append(fug_l_CO2 / 1e3)
            P_CO2_2_list.append(fug_l_CO2_2 / 1e3)
            alpha_range.append(alpha)
            T_range.append(T)
            etal.append(ηl)
            etav.append(ηv)
            a_assoc_l_list.append(a_assoc_l)
            a_assoc_v_list.append(a_assoc_v)
            a_res_l_list.append(a_res_l)
        except RuntimeWarning:
            P_CO2_list.append(np.nan)
            P_CO2_2_list.append(np.nan)
            alpha_range.append(np.nan)
            T_range.append(np.nan)
            etal.append(np.nan)
            etav.append(np.nan)
            a_assoc_l_list.append(np.nan)
            a_assoc_v_list.append(np.nan)
            a_res_l_list.append(np.nan)



    #%%
    plt.plot(α_data, P_CO2_data / 1e3, 'x', color=colors[i])
    plt.plot(α_range, P_CO2_list, '--', label=f'P_CO2 w assoc - T = {T}',  color=colors[i])
    plt.plot(α_range, P_CO2_2_list, ':', label=f'P_CO2_2 w no assoc - T = {T}',  color=colors[i])
    # plt.plot(α_range, P_CO2_2, ':', label=f'fug - T = {T}',  color=colors[i])

# data = np.column_stack((T_range, alpha_range, etal, etav, P_range, a_assoc_l_list, a_assoc_v_list, a_res_l_list))
# df = pd.DataFrame(data, columns=['T', 'alpha', 'ηl', 'ηv', 'P', 'a_assoc_l', 'a_assoc_v', 'a_res_l'])
df.to_csv('PC_SAFT_data_1.csv')
plt.xlabel("CO$_{2}$ Loading, mol CO$_{2}$/mol MEA", fontsize=16)
plt.ylabel("CO$_{2}$ pressure, kPa", fontsize=16)
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.yscale('log')
# plt.title(f'T = {T} C')
plt.legend()
plt.show()
# print(P_CO2_2)


