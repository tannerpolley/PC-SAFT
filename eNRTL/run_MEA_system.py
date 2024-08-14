import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import pyomo.environ as pyo

import idaes.logger as idaeslog
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale
from idaes.models_extra.column_models.properties import ModularPropertiesInherentReactionsInitializer
import idaes.core.util.scaling as iscale
from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
from MEA_eNRTL import get_prop_dict
from Parameter_Setup import create_and_scale_params_solve
from add_datasets import get_x
from PIL import Image
from matplotlib.pyplot import gca


def run_MEA_system():

    optarg = {
        # 'bound_push' : 1e-22,
        'nlp_scaling_method': 'user-scaling',
        'linear_solver': 'ma57',
        'OF_ma57_automatic_scaling': 'yes',
        'max_iter': 300,
        'tol': 1e-8,
        'constr_viol_tol': 1e-8,
        'halt_on_ampl_error': 'no',
        # 'mu_strategy': 'monotone',
    }
    init_outlevel = idaeslog.WARNING

    def Rochelle_fit(loading, T):
        return np.exp((39.3 - 12155 / T - 19.0 * loading ** 2 + 1105 * loading / T + 12800 * loading ** 2 / T)) / 1e3


    key_2 = [['tab:blue', 'o'],
             ['tab:orange', 's'],
             ['tab:green', '^'],
             ['tab:red', 'D'],
             ['tab:purple', 'p']]
    key = {}

    data_dir = r'C:\Users\Tanner\Documents\git\MEA\data\data_sets_to_load'
    i = 0
    for file in os.listdir(data_dir):
        if 'VLE' in file:
            name = file.split('_')[0]
            key[name] = key_2[i]
            i += 1

    temp_style = ['-', '--', ':']
    mec = ['black', 'dimgray', 'white']

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    if isinstance(axs, list):
        axs = axs[0]
    MAPE_model = []
    MAPE_Roch = []
    temperature = np.array([40, 80, 120])
    loading_min_constraint = .1
    loading_max_constraint = .6
    w_MEA = .3
    leg_loc = [.425, .675, .9]

    for i_t, t in enumerate(temperature):
        loading_min = []
        loading_max = []
        for file in os.listdir(data_dir):
            df = pd.read_csv(data_dir + '/' + file)
            if 'VLE' in file and (w_MEA in df['MEA_weight_fraction'].values and t in df['temperature'].values):
                df = df[(df['MEA_weight_fraction'] == w_MEA) &
                        (df['temperature'] == t) &
                        (df['CO2_loading'] > loading_min_constraint) &
                        (df['CO2_loading'] < loading_max_constraint)]
                CO2_loading = df['CO2_loading'].to_numpy()
                loading_min.append(min(CO2_loading))
                loading_max.append(max(CO2_loading))
        loading = np.linspace(min(loading_min), max(loading_max), 20)


        P_CO2_model = []
        for alpha in loading:
            m = pyo.ConcreteModel()
            create_and_scale_params_solve(m)
            m.state_block = m.params.build_state_block([0], has_phase_equilibrium=False, defined_state=True)
            blk = m.state_block[0]
            x_CO2, x_MEA, x_H2O, n_tot = get_x(alpha, w_MEA)
            blk.temperature.fix(t + 273.15)
            blk.pressure.fix(200000)
            blk.mole_frac_comp["H2O"].fix(x_H2O)
            blk.mole_frac_comp["MEA"].fix(x_MEA)
            blk.mole_frac_comp["CO2"].fix(x_CO2)

            iscale.calculate_scaling_factors(m)
            state_init = ModularPropertiesInherentReactionsInitializer(solver="ipopt",
                                                                       solver_options=optarg,
                                                                       output_level=init_outlevel)
            state_init.initialize(m.state_block)
            m_scaled = pyo.TransformationFactory('core.scale_model').create_using(m, rename=False)
            solver = get_solver("ipopt", options=optarg)
            solver.solve(m_scaled, tee=False)
            pyo.TransformationFactory('core.scale_model').propagate_solution(m_scaled, m)

            P_CO2_model.append(pyo.value(blk.fug_phase_comp["Liq", "CO2"]) / 1e3)

        P_CO2_model_interp = interp1d(loading, P_CO2_model, kind='cubic')
        P_CO2_Roch = Rochelle_fit(loading, t + 273.15)
        P_CO2_Roch_interp = interp1d(loading, P_CO2_Roch, kind='cubic')
        data_dir = r'C:\Users\Tanner\Documents\git\MEA\data\data_sets_to_load'
        counter = 0
        T = t

        MAPE_sum_model = []
        MAPE_sum_Roch = []
        for file in os.listdir(data_dir):
            name = file.split('_')[0]
            df = pd.read_csv(data_dir + '/' + file)
            print(name, len(df))

            if 'VLE' in file and (w_MEA in df['MEA_weight_fraction'].values and T in df['temperature'].values):
                df = df[(df['MEA_weight_fraction'] == w_MEA) &
                        (df['temperature'] == T) &
                        (df['CO2_loading'] > loading_min_constraint) &
                        (df['CO2_loading'] < loading_max_constraint)]
                loading_data = df['CO2_loading'].to_numpy()
                CO2_pressure_data = df['CO2_pressure'].to_numpy()
                axs.semilogy(loading_data, CO2_pressure_data,
                             label=f"{T} C - Data: {name}", linestyle="none",
                             marker=key[name][1], markersize=10, mfc=key[name][0], mec=mec[i_t])
                counter += 1
                Sigma_1 = 0
                Sigma_2 = 0
                for i, alpha in enumerate(loading_data):
                    Sigma_1 += abs((P_CO2_model_interp(alpha) - CO2_pressure_data[i]) / CO2_pressure_data[i])
                    Sigma_2 += abs((P_CO2_Roch_interp(alpha) - CO2_pressure_data[i]) / CO2_pressure_data[i])

                MAPE_sum_model.append(Sigma_1/len(loading_data))
                MAPE_sum_Roch.append(Sigma_2/len(loading_data))

        axs.semilogy(loading, P_CO2_Roch, linestyle=temp_style[i_t], color='r',
                     label=f"{T} C - Roch. Fit - MAPE: {np.mean(MAPE_sum_Roch):.2%}")
        axs.semilogy(loading, P_CO2_model, linestyle=temp_style[i_t], color='k',
                     label=f"{T} C - Model Fit - MAPE: {np.mean(MAPE_sum_model):.2%}")

    axs.set_xlabel("CO$_{2}$ Loading, mol CO$_{2}$/mol MEA", fontsize=16)
    axs.set_ylabel("CO$_{2}$ pressure, kPa", fontsize=16)
    axs.set_title(f"Model Fit for VLE CO2 Solubility at {w_MEA:.0%} MEA", fontsize=20)
    axs.tick_params(labelsize=14)
    axs.legend(loc='lower right', fontsize=10, ncol=2)

    # img = np.asarray(Image.open(r'C:\Users\Tanner\Documents\git\MEA\data\Parameters\Parameters.png'))
    # axs[1].imshow(img)
    # axs[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_MEA_system()
