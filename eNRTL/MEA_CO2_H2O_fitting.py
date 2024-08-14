import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import pyomo.environ as pyo

import idaes.logger as idaeslog
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale
from idaes.models_extra.column_models.properties import ModularPropertiesInherentReactionsInitializer

from Create_Param_Pic import create_param_pic
from Plotting import plot_VLE, plot_dH_abs
from Uncertainty_Analysis import uncertainty_analysis
from add_datasets import add_ABS_dataset, add_VLE_dataset, loss
from Parameter_Setup import get_estimated_params, create_and_scale_params
from run_MEA_system import run_MEA_system


# %% Model Setup

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

if __name__ == "__main__":

    logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)
    init_outlevel = idaeslog.WARNING
    m = pyo.ConcreteModel()
    create_and_scale_params(m)

    # %% Dataset Implementation

    obj_expr = 0
    dfs = []
    param_block_names = []
    dict_kim_dH_abs = {}

    VLE_dir = r"C:\Users\Tanner\Documents\git\MEA\data\data_sets_to_load"
    exclude_list = ['Xu']

    for name in os.listdir(VLE_dir):
        filename = VLE_dir + os.sep + name
        name, year, dataset_type = name.split('_')
        if name in exclude_list:
            continue
        dataset_type = dataset_type.split('.')[0]
        df = pd.read_csv(filename, index_col=None)
        dfs.append(df)
        param_block_name = name + '_' + dataset_type
        param_block_names.append(param_block_name)

        if dataset_type == 'dHabs':
            setattr(m, param_block_name, m.params.build_state_block(range(len(df) + len(df["temperature"].unique())),
                                                                    defined_state=True))
        elif dataset_type == 'VLE':
            setattr(m, param_block_name, m.params.build_state_block(range(len(df)), defined_state=True))

        param_block = getattr(m, param_block_name)
        if dataset_type == 'VLE':
            if 'total_pressure' in df.columns:
                obj_expr = add_VLE_dataset(param_block, df, obj_expr)
            else:
                obj_expr = add_VLE_dataset(param_block, df, obj_expr, has_total_pressure=False)
        elif dataset_type == 'dHabs':
            obj_expr, dict_kim_dH_abs = add_ABS_dataset(m, df, param_block, obj_expr)

    # %% Model Initializing and Solving

    iscale.calculate_scaling_factors(m)
    print(f"DOF: {degrees_of_freedom(m)}")
    state_init = ModularPropertiesInherentReactionsInitializer(solver="ipopt",
                                                               solver_options=optarg,
                                                               output_level=init_outlevel)
    for param_block_name in param_block_names:
        param_block = getattr(m, param_block_name)
        state_init.initialize(param_block)

    iscale.calculate_scaling_factors(m)
    estimated_vars = get_estimated_params(m)

    reg = .01
    obj_expr += sum([reg * loss((var - var.value) * iscale.get_scaling_factor(var)) for var in estimated_vars])

    m.obj = pyo.Objective(expr=obj_expr)
    m_scaled = pyo.TransformationFactory('core.scale_model').create_using(m, rename=False)
    m_scaled.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m_scaled.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    estimated_vars_scaled = [m_scaled.find_component(var.name) for var in estimated_vars]
    for var in estimated_vars_scaled:
        var.unfix()
    optarg.pop("nlp_scaling_method", None)  # Scaled model doesn't need user scaling
    solver = get_solver("ipopt", options=optarg)
    solver.solve(m_scaled, tee=False)
    pyo.TransformationFactory('core.scale_model').propagate_solution(m_scaled, m)

    # %% Uncertainty Analysis

    parameters, W_value = uncertainty_analysis(m_scaled, estimated_vars, estimated_vars_scaled)
    df = pd.DataFrame(parameters)
    df.to_csv(r'C:\Users\Tanner\Documents\git\MEA\data\Parameters\ParametersOG.csv', index=False)
    create_param_pic(df, f"Obj: {pyo.value(m.obj): .4f} - H: {W_value: .4f}")

    # %% Plotting

    for df, param_block_name in zip(dfs, param_block_names):
        param_block = getattr(m, param_block_name)
        if 'dHabs' in param_block_name:
            plot_dH_abs(param_block, dict_kim_dH_abs)
        elif 'VLE' in param_block_name:
            # print()
            plot_VLE(df, param_block)
    # img = np.asarray(Image.open(r'C:\Users\Tanner\Documents\git\MEA\data\Parameters\Parameters.png'))
    # plt.figure(figsize=(20, 20))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    # #
    run_MEA_system()
