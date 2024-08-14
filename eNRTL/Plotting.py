import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo

def plot_VLE(df, blk):
    colors = ["firebrick", "royalblue", "forestgreen", "goldenrod",
              "magenta", "orangered", "crimson", "darkcyan", "indigo"]
    P_CO2_rel_err = 0
    n_data = 0
    if blk.name == 'Xu_VLE':
        df['temperature'] = np.round(df['temperature'].to_numpy(), -1)
        df = df.sort_values(['temperature', 'CO2_loading'])

    index = [df['MEA_weight_fraction'], df['temperature']]
    df = pd.DataFrame(df.to_numpy()[:, 2:], index, columns=list(df.columns[2:]))
    count = 0
    n_subplots = len(df.index.unique('MEA_weight_fraction'))
    fig, axs = plt.subplots(n_subplots, figsize=(8, 8))
    fig.suptitle(f"Dataset: {blk.name}")
    count_2 = 0
    for i1, w_MEA in enumerate(df.index.unique('MEA_weight_fraction')):
        for j, T in enumerate(df.index.unique('temperature')):
            x = []
            y_data = []
            y_model = []
            for i, (_, row) in enumerate(df.loc[w_MEA, T].iterrows()):
                n_data += 1
                CO2_loading = row["CO2_loading"]
                x.append(CO2_loading)
                y_data.append(row["CO2_pressure"])

                fug_CO2 = pyo.value(blk[count].fug_phase_comp["Liq", "CO2"]) / 1e3
                # fug_CO2 = row["CO2_pressure"]

                y_model.append(fug_CO2)
                count += 1
            if n_subplots > 1:
                axs[i1].semilogy(x, y_data, linestyle="none", marker="o", color=colors[count_2])
                axs[i1].semilogy(x, y_model, linestyle="--", color=colors[count_2], label=f"T = {T}")
            else:
                axs.semilogy(x, y_data, linestyle="none", marker="o", color=colors[count_2])
                axs.semilogy(x, y_model, linestyle="--", color=colors[count_2], label=f"T = {T}")
            count_2 += 1

            for y_data_pt, y_model_pt in zip(y_data, y_model):
                P_CO2_rel_err += abs((y_data_pt - y_model_pt) / y_data_pt)

        if n_subplots > 1:
            axs[i1].set_xlabel("CO2 Loading")
            axs[i1].set_ylabel("CO2 vapor pressure (kPa)")
            axs[i1].set_title(f"{w_MEA:.0%} MEA - Rel Error: {P_CO2_rel_err / n_data:.4f}")
            axs[i1].legend()
        else:
            axs.set_xlabel("CO2 Loading")
            axs.set_ylabel("CO2 vapor pressure (kPa)")
            axs.set_title(f"{w_MEA:.0%} MEA - Rel Error: {P_CO2_rel_err / n_data:.4f}")
            axs.legend()

    plt.tight_layout()
    fig.show()


def plot_dH_abs(blk, dict_dH_abs):

    colors = ["firebrick", "royalblue", "forestgreen", "goldenrod",
              "magenta", "orangered", "crimson", "darkcyan"]
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111)
    ax.set_title('Heat of Absorption',
                  fontsize=20,
                  fontweight='bold')
    # ax.set_ylabel('Excess Enthalpy, kJ/mol', fontsize=20)
    # ax.set_xlabel('Loading, mol CO$_{2}$/mol MEA', fontsize=20)

    def rochelle(loading):
        R = 8.314

        return -R * (-12155 + 1105 * loading + 12800 * loading ** 2) * 1e-3

    x_roch = np.linspace(.05, .4, 101)
    y_Rochelle = rochelle(x_roch)

    for i, T in enumerate(dict_dH_abs.keys()):
        subdict = dict_dH_abs[T]

        x = []
        y_data = []
        y_model = []
        y_model_2 = []

        for k, loading in enumerate(subdict["CO2_loading"]):
            if loading <= .8 and (subdict["dH_abs"][k] <= 1300):
                # print(pyo.value(subdict["enth_abs_expr"][k]))
                x.append(loading)
                y_data.append(subdict["dH_abs"][k])
                y_model.append(1e-3 * pyo.value(subdict["enth_abs_expr"][k]))
                # y_model_2.append(1e-3 * pyo.value(subdict["enth_abs_expr_2"][k]))
                # y_model.append(1e-3 * pyo.value(subdict["excess_enth_list"][k]))
                # y_model_2.append(1e-3 * pyo.value(subdict["enth_list_2"][k]))
                # print(pyo.value(subdict["enth_abs_expr"][k]) - pyo.value(subdict["enth_abs_expr_2"][k]))
                # print(pyo.value(blk[k].enth_mol_phase['Liq']), pyo.value(blk[k].energy_internal_mol_phase['Liq']))
        # print(np.array(y_model)-np.array(y_model_2))
        ax.plot(x, y_data, linestyle="none", marker="o", color=colors[i])
        ax.plot(x, y_model, linestyle="--", color=colors[i], label=f"T = {T}")
        # ax.plot(x, y_model_2, linestyle="-.", color=colors[i], label=f"T = {T}")
    ax.plot(x_roch, y_Rochelle, linestyle=":", color='black', label=f"Rochelle")
    # ax.set_xlabel("CO2 Loading")
    ax.set_ylabel("Enth_Abs (kJ/mol)")
    # ax.set_ylabel('Excess Enthalpy, kJ/mol',fontsize=20)
    ax.set_xlabel('Loading, mol CO$_{2}$/mol MEA',fontsize=20)
    ax.tick_params(labelsize=20, direction='in')
    plt.tight_layout()
    ax.legend(fontsize=20)
    plt.show()