import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pcsaft import flashTQ
from scipy.interpolate import interp1d
from Get_True_Mol_Frac import get_true_mol_frac
from PC_SAFT import flash, PCSAFT
from PC_SAFT_v2 import flash_v2
from PC_SAFT_v3 import flash_v3
from pcsaft_electrolyte import *
# from pcsaft import flashTQ, pcsaft_ares, pcsaft_den, pcsaft_p, pcsaft_fugcoef
import time

#%%
plt.figure(figsize=(10, 10))


def Rochelle_fit(loading, T):
    return np.exp((39.3 - 12155 / T - 19.0 * loading ** 2 + 1105 * loading / T + 12800 * loading ** 2 / T)) / 1e3


interp_data = pd.read_csv(r"C:\Users\Tanner\Documents\git\eNRTL_Fitting_Routine\compare_data_without_eNRTL.csv")

colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown']
for i, T in enumerate([40, 60, 80, 100, 120]):
    df = pd.read_csv(r'data/data_sets_to_load/Jou_1995_VLE.csv')
    df = df[(df['temperature'] == T) &
            (df['CO2_loading'] > .1) &
            (df['CO2_loading'] < .6)]
    P_CO2_data, α_data = df['CO2_pressure'].to_numpy(), df['CO2_loading'].to_numpy()
    w_MEA = .3
    Tl = 273.15 + T
    P_CO2_list = []
    P_CO2_2_list = []
    P_CO2_3_list = []
    P_CO2_4_list = []
    loading_range = np.linspace(α_data[0], α_data[-1], 21)
    interp_data_cut = interp_data[interp_data['temperature'] == T]
    P_interp = interp1d(interp_data_cut['loading'], interp_data_cut['Pressure'])
    P_H2O_interp = interp1d(interp_data_cut['loading'], interp_data_cut['fug_H2O'])
    print(T)
    for loading in loading_range:
        x = get_true_mol_frac(loading, w_MEA, Tl)[:3]
        x = np.array([xi / sum(x) for xi in x])
        Pg = float(P_interp(loading)) * 1e3
        P_CO2_roch = Rochelle_fit(loading, T + 273.15) * 1e3
        y_CO2_g = P_CO2_roch / Pg
        # psat_H2O = np.exp(73.649 + -7258.2 / Tl + -7.3037 * np.log(Tl) + 4.1653e-6 * Tl ** 2)
        y_H2O_g = P_H2O_interp(loading)*1e3 / Pg
        yg = [y_CO2_g, 1 - y_H2O_g - y_CO2_g, y_H2O_g]

        sigma_H2O = 2.7927 + (10.11 * np.exp(-.01775 * T - 1.417 * np.exp(-.01146 * T)))

        prop_dic_2 = {
            'm': np.array([2.079, 3.0353, 1.9599]),
            's': np.array([2.7852, 3.0435, 2.363]),
            'e': np.array([169.21, 277.174, 279.42]),
            'vol_a': np.array([0, .037470, .1750]),
            'e_assoc': np.array([0, 2586.3, 2059.28]),
            'k_ij': np.array([[0.0, .16, .065],
                              [.16, 0.0, -.18],
                              [.065, -.18, 0.0]]),
            'dielc': 75
        }

        # start_time = time.time()
        # try:
        #     # mix_l = PCSAFT(Tl, x,
        #     #                prop_dic_2, P_sys=163754.41812246613)
        #     #
        #     # rho = mix_l.v() ** -1
        #     # print('Mine')
        #     # print('rho', rho)
        #     # print('a_res', mix_l.a_res())
        #     P, y = flash(x, yg, Tl, Pg, prop_dic_2, flash_type='Bubble_P')
        #     P_CO2 = P*y[0]
        #     P_CO2_list.append(P_CO2/1e3)
        #     print(f'Pressure - Guess: {Pg/1e3:.2f}, Actual: {P/1e3:.2f}')
        #     print(f'y_CO2: Guess: {yg[0]:.3e}, Actual: {y[0]:3e}')
        #     print(f'y_H2O: Guess: {yg[2]:.3e}, Actual: {y[2]:3e}')
        #     print()
        #
        #     # print('P_CO2', P_CO2)
        # except RuntimeWarning:
        #     P_CO2_list.append(np.nan)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print("Execution time:", execution_time, "seconds")
        # print()

        # start_time = time.time()
        # try:
        #     m = prop_dic_2['m']
        #     σ = prop_dic_2['s']
        #     ϵ_k = prop_dic_2['e']
        #     κ_AB = prop_dic_2['vol_a']
        #     ϵ_AB_k = prop_dic_2['e_assoc']
        #     k_ij = prop_dic_2['k_ij']
        #     rho = pcsaft_den(x=x, m=m, s=σ, e=ϵ_k, t=Tl, p=163754.41812246613,
        #                      k_ij=k_ij, e_assoc=ϵ_AB_k, vol_a=κ_AB)
        #     print('zmeri')
        #     # print('rho', rho)
        #     # a_res = pcsaft_ares(x=x, m=m, s=σ, e=ϵ_k, t=Tl, rho=rho,
        #     #                   k_ij=k_ij, e_assoc=ϵ_AB_k, vol_a=κ_AB)
        #     # print('a_res', a_res)
        #     P, y = pcsaft_bubbleP(p_guess=Pg, xv_guess=yg, x=x, m=m, s=σ, e=ϵ_k, t=Tl,
        #                           k_ij=k_ij, e_assoc=ϵ_AB_k, vol_a=κ_AB)
        #     P_CO2_2 = P*y[0]
        #     P_CO2_2_list.append(P_CO2_2)
        #     print('P_CO2', P_CO2_2)
        #
        # except ValueError:
        #     P_CO2_2_list.append(np.nan)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # # print("Execution time:", execution_time, "seconds")
        # print()

        prop_dic_2 = {
            'm': np.array([2.079, 3.0353, 1.9599, 0, 0, 0]),
            's': np.array([2.7852, 3.0435, 2.363, 0, 0, 0]),
            'e': np.array([169.21, 277.174, 279.42, 0, 0, 0]),
            'vol_a': np.array([0, .037470, .1750, 0, 0, 0]),
            'e_assoc': np.array([0, 2586.3, 2059.28, 0, 0, 0]),
            'k_ij': np.array([[0.0, .16, .065, 0, 0, 0],
                              [.16, 0.0, -.18, 0, 0, 0],
                              [.065, -.18, 0.0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0]]),
            'dielc': 75,
            'z': np.array([0, 0, 0, +1, -1, -1])
        }

        start_time = time.time()
        try:

            # print('zmeri')
            P, xl, y = flashTQ(t=Tl, q=0, x=x, params=prop_dic_2, p_guess=Pg)
            # print(P, xl, y)
            P_CO2_2 = P * y[0] / 1e3
            P_CO2_2_list.append(P_CO2_2)
            print(f'Pressure - Guess: {Pg/1e3:.2f}, Actual: {P/1e3:.2f}')
            print(f'y_CO2: Guess: {yg[0]:3e}, Actual: {y[0]:3e}')
            print(f'y_H2O: Guess: {yg[2]:3e}, Actual: {y[2]:3e}')
            print()

        except:
            P_CO2_2_list.append(np.nan)
        end_time = time.time()
        execution_time = end_time - start_time
        # print("Execution time:", execution_time, "seconds")
        # print()

    #%%
    plt.plot(α_data, P_CO2_data, 'x', color=colors[i])
    # plt.plot(loading_range, P_CO2_list, 'o', label=f'Mine - T = {T}', color=colors[i])
    plt.plot(loading_range, P_CO2_2_list, ':', label=f'Zmeri - T = {T}', color=colors[i])
    # plt.plot(loading_range, Rochelle_fit(loading_range, Tl), ':', label=f'Roch - T = {T}', color=colors[i])

plt.xlabel("CO$_{2}$ Loading, mol CO$_{2}$/mol MEA", fontsize=16)
plt.ylabel("CO$_{2}$ pressure, kPa", fontsize=16)
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.yscale('log')
plt.legend()
plt.show()

# prop_dic = {'m': np.array([2.079, 3.0353, 1.2046]),
#             's': np.array([2.7852, 3.0435, sigma_H2O]),
#             'e': np.array([169.21, 277.174, 353.94]),
#             # 'vol_a': np.array([0, .037470, .04509]),
#             # 'e_assoc': np.array([0, 2586.3, 2425.67]),
#             'vol_a': np.array([0, 0, 0]),
#             'e_assoc': np.array([0, 0, 0]),
#             'k_ij': np.array([[0.0, .16, .065],
#                               [.16, 0.0, -.18],
#                               [.065, -.18, 0.0]])
#             }

# prop_dic_2 = {'m': np.array([2.079, 3.0353, 1.2046]),
#             's': np.array([2.7852, 3.0435, sigma_H2O]),
#             'e': np.array([169.21, 277.174, 353.94]),
#             'vol_a': np.array([0, .037470, .04509]),
#             'e_assoc': np.array([0, 2586.3, 2425.67]),
#             'k_ij': np.array([[0.0, .16, .065],
#                               [.16, 0.0, -.18],
#                               [.065, -.18, 0.0]])
#             }
