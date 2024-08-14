import sys, os, logging
sys.path.append('../')


import pandas as pd
import numpy as np
from scipy.linalg import svd, schur, eigh
import matplotlib.pyplot as plt

import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.interior_point.inverse_reduced_hessian import inv_reduced_hessian_barrier

import idaes.logger as idaeslog
from idaes.core import Solvent
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.initialization import solve_indexed_blocks
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.pure import NIST
from CESAR_1_eNRTL import get_prop_dict, create_heat_capacity_no_inherent_rxns

def loss(x):
    return 0.5*x ** 2

def get_estimated_params(m, henry):
    param_list = [
        m.params.Liq.tau_A["H2O", "PZ"],
        m.params.Liq.tau_B["H2O", "PZ"],
        m.params.Liq.tau_A["PZ", "H2O"],
        m.params.Liq.tau_B["PZ", "H2O"],
        m.params.Liq.alpha["H2O", "PZ"],
    ]
    if henry:
        param_list += [
            m.params.PZ.henry_coeff_1,
            # m.params.PZ.henry_coeff_2,
            m.params.PZ.henry_coeff_3,
            m.params.PZ.henry_coeff_4
        ]
    return param_list

def create_and_scale_params(m, henry):
    config = get_prop_dict(["H2O", "PZ"], excluded_rxns=["H2O_autoionization", "PZ_protonation"])

    params = m.params = GenericParameterBlock(**config)

    gsf = iscale.get_scaling_factor
    scaling_factor_flow_mol = 1/100
    params.set_default_scaling("enth_mol_phase", 3e-4)
    params.set_default_scaling("pressure", 1e-5)
    params.set_default_scaling("temperature", 1)
    params.set_default_scaling("flow_mol", scaling_factor_flow_mol)
    params.set_default_scaling("flow_mol_phase", scaling_factor_flow_mol)

    params.set_default_scaling("flow_mass_phase", scaling_factor_flow_mol / 18e-3)  # MW mixture ~= 24 g/Mol
    params.set_default_scaling("dens_mol_phase", 1 / 18000)
    params.set_default_scaling("visc_d_phase", 700)
    params.set_default_scaling("log_k_eq", 1)

    mole_frac_scaling_factors = {
        "H2O": 2,
        "PZ": 2,
    }
    mole_frac_true_scaling_factors = {
        "PZ": 2,
        "H2O": 2,
        # "HCO3^-": 5e5,
        # "H3O^+": 5e5,
        # "CO3^2-": 1e12,
        # "OH^-": 5e11,
        # "HCO3^-": 1e3,
        "H3O^+": 1e11,
        # "CO3^2-": 1e3,
        "OH^-": 1e3,
        "PZH^+": 1e3
    }
    for comp, sf_x in mole_frac_scaling_factors.items():
        params.set_default_scaling("mole_frac_comp", sf_x, index=comp)
        params.set_default_scaling("mole_frac_phase_comp", sf_x, index=("Liq", comp))
        params.set_default_scaling(
            "flow_mol_phase_comp",
            sf_x * scaling_factor_flow_mol,
            index=("Liq", comp)
        )

    # for comp, sf_x in mole_frac_true_scaling_factors.items():
    #     params.set_default_scaling("mole_frac_phase_comp_true", sf_x, index=("Liq", comp))
    #     params.set_default_scaling(
    #         "flow_mol_phase_comp_true",
    #         sf_x * scaling_factor_flow_mol,
    #         index=("Liq", comp)
    #     )

    # params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1e4, index="H2O_autoionization")
    # params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1e4, index="PZ_protonation")

    iscale.set_scaling_factor(m.params.Liq.alpha, 1) 
    iscale.set_scaling_factor(m.params.Liq.tau_A, 1) # Reminder that it's well-scaled
    iscale.set_scaling_factor(m.params.Liq.tau_B, 1/300)

    if henry:
        iscale.set_scaling_factor(m.params.PZ.henry_coeff_1, 1/300)
        iscale.set_scaling_factor(m.params.PZ.henry_coeff_2, 1) # Reminder that it's well-scaled
        iscale.set_scaling_factor(m.params.PZ.henry_coeff_3, 300)
        iscale.set_scaling_factor(m.params.PZ.henry_coeff_4, 1) # Reminder that it's well-scaled

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
    init_outlevel = idaeslog.INFO
    henry=False

    data = os.sep.join([this_file_dir(), "data"])
    m = pyo.ConcreteModel()
    create_and_scale_params(m, henry=henry)

    m.params.Liq.tau_A["H2O", "PZ"].set_value(5.32)
    m.params.Liq.tau_B["H2O", "PZ"].set_value(-1279.99)
    m.params.Liq.tau_A["PZ", "H2O"].set_value(-0.6)
    m.params.Liq.tau_B["PZ", "H2O"].set_value(-698.51)
    m.params.Liq.alpha["H2O", "PZ"].set_value(0.3)
    # m.params.Liq.tau_A["H2O", "PZ"].set_value(4.95)
    # m.params.Liq.tau_B["H2O", "PZ"].set_value(-1315.65)
    # m.params.Liq.tau_A["PZ", "H2O"].set_value(0.95)
    # m.params.Liq.tau_B["PZ", "H2O"].set_value(-1208.62)
    # m.params.Liq.alpha["H2O", "PZ"].set_value(0.3)

    obj_expr = 0

    df_hartono = pd.read_csv(os.sep.join([data, "hartono_et_al_2013_PZ_T_x_y.csv"]), index_col=None)

    n_data_hartono = len(df_hartono["temperature"])
    m.state_hartono = m.params.build_state_block(range(n_data_hartono), defined_state=True)

    for i, row in df_hartono.iterrows():
        m.state_hartono[i].flow_mol.fix(100)
        m.state_hartono[i].mole_frac_comp["H2O"].fix(1 - row["mole_frac_liq_PZ"])
        m.state_hartono[i].mole_frac_comp["PZ"].fix(row["mole_frac_liq_PZ"])
        m.state_hartono[i].pressure.fix(row["pressure"]*1e3) # Pressure in kPa
        m.state_hartono[i].temperature.fix(row["temperature"] + 273.15) # Temperature in C
        obj_expr +=  (
            # Total pressure
            loss(
                pyo.log((m.state_hartono[i].fug_phase_comp["Liq","PZ"] + m.state_hartono[i].fug_phase_comp["Liq","H2O"])/pyo.units.Pa) 
                - pyo.log(row["pressure"]*1e3)
            )
            # PZ partial pressure
            + loss(
                m.state_hartono[i].log_fug_phase_comp["Liq","PZ"] 
                - pyo.log(row["mole_frac_vap_PZ"] * row["pressure"]*1e3)
            )
        )

    df_nguyen_3_3_1 = pd.read_csv(os.sep.join([data, "nguyen_dissertation_2013_PZ_H2O_pressure_table_3_3_1.csv"]), index_col=None)

    n_data_nguyen_3_3_1 = len(df_nguyen_3_3_1["temperature"])
    m.state_nguyen_3_3_1 = m.params.build_state_block(range(n_data_nguyen_3_3_1), defined_state=True)
 
    molality=0.48
    n_tot = molality+1/0.01802
    x_PZ = molality/n_tot
    x_H2O = 1 - x_PZ

    for i, row in df_nguyen_3_3_1.iterrows():

        m.state_nguyen_3_3_1[i].flow_mol.fix(100)
        m.state_nguyen_3_3_1[i].mole_frac_comp["H2O"].fix(x_H2O)
        m.state_nguyen_3_3_1[i].mole_frac_comp["PZ"].fix(x_PZ)
        m.state_nguyen_3_3_1[i].pressure.fix(row["pressure_PZ"]+row["pressure_H2O"]*1e3) # H2O Pressure in kPa, PZ in Pa
        m.state_nguyen_3_3_1[i].temperature.fix(row["temperature"] + 273.15) # Temperature in C
        obj_expr +=  (
            # H2O partial pressure
            loss(
                m.state_nguyen_3_3_1[i].log_fug_phase_comp["Liq","H2O"] 
                - pyo.log(row["pressure_H2O"]*1e3)
            )
            # PZ partial pressure
            + loss(
                m.state_nguyen_3_3_1[i].log_fug_phase_comp["Liq","PZ"] 
                - pyo.log(row["pressure_PZ"])
            )
        )

    df_nguyen_5_3_3 = pd.read_csv(os.sep.join([data, "nguyen_dissertation_2013_PZ_H2O_pressure_table_5_3_3.csv"]), index_col=None)

    n_data_nguyen_5_3_3 = len(df_nguyen_5_3_3["temperature"])
    m.state_nguyen_5_3_3 = m.params.build_state_block(range(n_data_nguyen_5_3_3), defined_state=True)
 
    molality=8.0
    n_tot = molality+1/0.01802
    x_PZ = molality/n_tot
    x_H2O = 1 - x_PZ

    for i, row in df_nguyen_5_3_3.iterrows():

        m.state_nguyen_5_3_3[i].flow_mol.fix(100)
        m.state_nguyen_5_3_3[i].mole_frac_comp["H2O"].fix(x_H2O)
        m.state_nguyen_5_3_3[i].mole_frac_comp["PZ"].fix(x_PZ)
        m.state_nguyen_5_3_3[i].pressure.fix(row["pressure_PZ"]+row["pressure_H2O"]*1e3) # H2O Pressure in kPa, PZ in Pa
        m.state_nguyen_5_3_3[i].temperature.fix(row["temperature"] + 273.15) # Temperature in C
        obj_expr +=  (
            # H2O partial pressure
            loss(
                m.state_nguyen_5_3_3[i].log_fug_phase_comp["Liq","H2O"] 
                - pyo.log(row["pressure_H2O"]*1e3)
            )
            # PZ partial pressure
            + loss(
                m.state_nguyen_5_3_3[i].log_fug_phase_comp["Liq","PZ"] 
                - pyo.log(row["pressure_PZ"])
            )
        )

    df_hillard_9_2_1 = pd.read_csv(os.sep.join([data, "hillard_2008_figure_9_2_1.csv"]), index_col=None)

    n_data_hillard_9_2_1 = len(df_hillard_9_2_1["temperature"])
    m.state_hillard_9_2_1 = m.params.build_state_block(range(n_data_hillard_9_2_1), defined_state=True)

    for i, row in df_hillard_9_2_1.iterrows():
        weight = 1
        x_PZ = float(row["x_PZ"])
        if x_PZ <1e-3:
            x_PZ = 1e-3
        if x_PZ > 0.3:
            weight = 5
        if x_PZ > 0.6:
            weight = 20
        m.state_hillard_9_2_1[i].flow_mol.fix(100)
        m.state_hillard_9_2_1[i].mole_frac_comp["H2O"].fix(1-x_PZ)
        m.state_hillard_9_2_1[i].mole_frac_comp["PZ"].fix(x_PZ)
        m.state_hillard_9_2_1[i].pressure.fix(row["P_tot"]*1e3) # Pressure in kPa
        m.state_hillard_9_2_1[i].temperature.fix(row["temperature"] + 273.15) # Temperature in C
        obj_expr +=  (
            # Total Pressure
            weight*loss(
                pyo.log((m.state_hillard_9_2_1[i].fug_phase_comp["Liq","PZ"] + m.state_hillard_9_2_1[i].fug_phase_comp["Liq","H2O"])/pyo.units.Pa) 
                - pyo.log(row["P_tot"]*1e3)
            )
        )

    df_hillard_9_2_2 = pd.read_csv(os.sep.join([data, "hillard_2008_figure_9_2_2.csv"]), index_col=None)

    n_data_hillard_9_2_2 = len(df_hillard_9_2_2["temperature"])
    m.state_hillard_9_2_2= m.params.build_state_block(range(n_data_hillard_9_2_2), defined_state=True)

    for i, row in df_hillard_9_2_2.iterrows():
        weight = 1
        molality_PZ = float(row["molality_PZ"])
        n_tot = molality_PZ+1/0.01802
        x_PZ = molality_PZ/n_tot
        x_H2O = 1 - x_PZ

        m.state_hillard_9_2_2[i].flow_mol.fix(100)
        m.state_hillard_9_2_2[i].mole_frac_comp["H2O"].fix(1-x_PZ)
        m.state_hillard_9_2_2[i].mole_frac_comp["PZ"].fix(x_PZ)
        # We don't have total pressure measurements for this experiment.
        # Since we don't have a Poynting term, PZ volatility and activity
        # shouldn't depend on total pressure, so fixe to constant value
        m.state_hillard_9_2_2[i].pressure.fix(2e5) 
        m.state_hillard_9_2_2[i].temperature.fix(row["temperature"] + 273.15) # Temperature in C
        obj_expr +=  (
            loss(
                m.state_hillard_9_2_2[i].log_fug_phase_comp["Liq","PZ"] 
                - pyo.log(row["pressure_PZ"]*1e3) # Pressure in kPa
            )
        )

    m.state_test = m.params.build_state_block(range(1,100), defined_state=True)

    for i in range(1, 100):
        x_PZ = i/100
        m.state_test[i].flow_mol.fix(100)
        m.state_test[i].mole_frac_comp["H2O"].fix(1-x_PZ)
        m.state_test[i].mole_frac_comp["PZ"].fix(x_PZ)
        m.state_test[i].pressure.fix(2e5)
        m.state_test[i].temperature.fix(113 + 273.15)

    df_chen_54_59  = pd.read_csv(os.sep.join([data, "chen_et_al_2010_54_59_table_2.csv"]), index_col=None)
    n_data_chen_54_59 = len(df_chen_54_59["temperature"])
    m.state_chen_54_59 = m.params.build_state_block(range(n_data_chen_54_59), defined_state=True)

    molality=8.0
    n_tot = molality+1/0.01802


    for i, row in df_chen_54_59.iterrows():
        x_PZ = float(row["x_PZ"])
        x_H2O = 1 - x_PZ
        m.state_chen_54_59[i].flow_mol.fix(100)
        m.state_chen_54_59[i].mole_frac_comp["H2O"].fix(x_H2O)
        m.state_chen_54_59[i].mole_frac_comp["PZ"].fix(x_PZ)
        m.state_chen_54_59[i].pressure.fix(2e5) # No pressure listed in experimental paper
        m.state_chen_54_59[i].temperature.fix(float(row["temperature"])) # Temperature in K

    print(f"DOF: {degrees_of_freedom(m)}")

    m.state_hartono.initialize(
        hold_state=False,
        outlvl=init_outlevel,
        optarg=optarg
    )
    m.state_nguyen_3_3_1.initialize(
        hold_state=False,
        outlvl=init_outlevel,
        optarg=optarg
    )
    m.state_nguyen_5_3_3.initialize(
        hold_state=False,
        outlvl=init_outlevel,
        optarg=optarg
    )

    m.state_hillard_9_2_1.initialize(
        hold_state=False,
        outlvl=init_outlevel,
        optarg=optarg
    )
    m.state_hillard_9_2_2.initialize(
        hold_state=False,
        outlvl=init_outlevel,
        optarg=optarg
    )

    m.state_test.initialize(
        hold_state=False,
        outlvl=init_outlevel,
        optarg=optarg
    )
    m.state_chen_54_59.initialize(
        hold_state=False,
        outlvl=init_outlevel,
        optarg=optarg
    )
    # Need to create heat capacity equations after initialization because
    # initialization method doesn't know about them
    create_heat_capacity_no_inherent_rxns(m.state_chen_54_59)
    assert degrees_of_freedom(m.state_chen_54_59) == 0
    for i, row in df_chen_54_59.iterrows():
        obj_expr +=  (
            # H2O partial pressure
            loss(
                0.1*(m.state_chen_54_59[i].Liq_cp - float(row["cp"]))
            )
        )

    iscale.calculate_scaling_factors(m)

    estimated_vars = get_estimated_params(m, henry=henry)
    # Apparently the model reduces to the Margules model at alpha=0,
    # which corresponds to random mixing. Alpha values less than zero
    # are not thermodynamically meaningful, because you can't get more
    # random than fully random.
    m.params.Liq.alpha["H2O", "PZ"].lb = 0
    # m.params.Liq.alpha["H2O", "PZ"].ub = 1

    # for var in estimated_vars:
    #     obj_expr += 0.01 * loss(
    #         (var - var.value) * iscale.get_scaling_factor(var)
    #     )

    m.obj = pyo.Objective(expr=obj_expr)
    m_scaled = pyo.TransformationFactory('core.scale_model').create_using(m, rename=False)
    optarg.pop("nlp_scaling_method", None) # Scaled model doesn't need user scaling
    optarg["max_iter"] = 500
    solver = get_solver(
        "ipopt",
        options=optarg
    )
    # solve_indexed_blocks(solver, m.state_chen_54_59, tee=True)
    estimated_vars_scaled = get_estimated_params(m_scaled, henry=henry)
    for var in estimated_vars_scaled:
        var.unfix()
    res = solver.solve(m_scaled, tee=True)
    pyo.assert_optimal_termination(res)

    


    inv_red_hess = inv_reduced_hessian_barrier(m_scaled, estimated_vars_scaled, tee=True, solver_options=optarg)
    W, V = eigh(inv_red_hess[1])

    print(f"Largest eigenvalue of inverse reduced Hessian: {W[-1]}")
    print("\n" + "Variables with most uncertainty:")
    for i in np.where(abs(V[:, -1]) > 0.3)[0]:
        print(str(i) + ": " + estimated_vars_scaled[i].name)

    pyo.TransformationFactory('core.scale_model').propagate_solution(m_scaled,m)
    print("==================================================")
    print("========== Variables with uncertainty ============")
    print("==================================================")
    def gsf(var):
        return iscale.get_scaling_factor(var, default=1)
    for i, var in enumerate(estimated_vars):
        print(f"{var.name}: {var.value:.3e} +/- {pyo.sqrt(inv_red_hess[1][i][i])/gsf(var):.3e}")
        

    fig = plt.figure()
    fig2 = plt.figure()
    ax = fig.subplots()
    ax2 = fig2.subplots()
    colors1 = ["firebrick", "royalblue", "forestgreen"]
    colors2 = ["gold", "mediumturquoise", "darkorchid"]
    print ("\n Data source: Hartono")
    P_tot_rel_err = 0
    P_PZ_rel_err = 0
    n_data = 0
    for T, color1, color2 in zip([60, 80, 100], colors1, colors2):
        x = []
        P_tot_data = []
        P_tot_model = []
        P_PZ_data = []
        P_PZ_model = []
        for i, row in df_hartono.loc[round(df_hartono["temperature"]) ==  T].iterrows():
            n_data+=1
            x_PZ = float(row["mole_frac_liq_PZ"])
            x.append(x_PZ)
            P_tot_data.append(row["pressure"])
            P_tot_model.append(
                pyo.value(m.state_hartono[i].fug_phase_comp["Liq","PZ"] 
                 + m.state_hartono[i].fug_phase_comp["Liq","H2O"]) / 1e3 # Convert from Pa to kPa
                )
            P_PZ_data.append(row["mole_frac_vap_PZ"] * row["pressure"] * 1e3)
            P_PZ_model.append(pyo.value(m.state_hartono[i].fug_phase_comp["Liq","PZ"]))

        for y_data, y_model in zip(P_tot_data,P_tot_model):
            P_tot_rel_err += abs((y_data-y_model)/y_data)
        for y_data, y_model in zip(P_PZ_data,P_PZ_model):
            P_PZ_rel_err += abs((y_data-y_model)/y_data)

        ax.semilogy(x, P_tot_data, linestyle="none", marker="o", color=color1)
        ax.semilogy(x, P_tot_model, linestyle="-", marker="none", color=color1, label=f"T = {T}" )
        ax2.semilogy(x, P_PZ_data, linestyle="none", marker="o", color=color2)
        ax2.semilogy(x, P_PZ_model, linestyle="-", marker="none", color=color2, label=f"T = {T}" )

   
    print(f"Total number of data points: {n_data}")
    print(f"Average relative error in total pressure: {P_tot_rel_err/n_data}")
    print(f"Average relative error in PZ pressure: {P_PZ_rel_err/n_data}")
    ax.set_xlabel("Piperizine mole fraction")
    ax2.set_xlabel("Piperizine mole fraction")
    ax.set_ylabel("Total vapor pressure (kPa)")
    ax2.set_ylabel("PZ vapor pressure (Pa)")
    ax.set_title("Hartono PZ-H2O Volatility")
    ax2.set_title("Hartono PZ Volatility")
    ax.legend()
    ax2.legend()

    fig.show()
    fig2.show()


    fig = plt.figure()
    ax = fig.subplots()
    colors = ["firebrick", "royalblue", "forestgreen"]
    print ("\n Data source: Hillard")
    P_tot_rel_err = 0
    n_data = 0
    for T, color in zip([113, 120, 199], colors):
        x = []
        y_data =[]
        y_model = []
        for i, row in df_hillard_9_2_1.loc[df_hillard_9_2_1["temperature"] ==  T].iterrows():
            n_data +=1
            x_PZ = float(row["x_PZ"])
            x.append(x_PZ)
            y_data.append(row["P_tot"])
            y_model.append(
                pyo.value(m.state_hillard_9_2_1[i].fug_phase_comp["Liq","PZ"] 
                 + m.state_hillard_9_2_1[i].fug_phase_comp["Liq","H2O"]) / 1e3 # Convert from Pa to kPa
                )
        for y_data_pt, y_model_pt in zip(y_data, y_model):
            P_tot_rel_err += abs((y_data_pt-y_model_pt)/y_data_pt) 
        ax.semilogy(x, y_data, linestyle="none", marker="o", color=color)
        ax.semilogy(x, y_model, linestyle="-", marker="none", color=color, label=f"T = {T}" )
    
    print(f"Total number of data points: {n_data}")
    print(f"Average relative error in total pressure: {P_tot_rel_err/n_data}")
    ax.set_xlabel("Piperizine mole fraction")
    ax.set_ylabel("Total vapor pressure (kPa)")
    ax.set_title("Hillard PZ-H2O VLE Data")
    ax.legend()

    fig.show()

    fig = plt.figure()
    ax = fig.subplots()
    colors = ["firebrick", "royalblue", "forestgreen", "gold", "darkorchid"]
    print ("\n Data source: Hillard")
    P_PZ_rel_err = 0
    n_data = 0
    for molality_PZ, color in zip(df_hillard_9_2_2["molality_PZ"].unique(), colors):
        x = []
        y_data =[]
        y_model = []
        for i, row in df_hillard_9_2_2.loc[df_hillard_9_2_2["molality_PZ"] ==  molality_PZ].iterrows():
            n_data += 1
            x.append(float(row["temperature"]))
            y_data.append(row["pressure_PZ"]*1e3)
            y_model.append(pyo.value(m.state_hillard_9_2_2[i].fug_phase_comp["Liq","PZ"]))
        
        for y_data_pt, y_model_pt in zip(y_data, y_model):
            P_PZ_rel_err += abs((y_data_pt-y_model_pt)/y_data_pt) 

        ax.semilogy(x, y_data, linestyle="none", marker="o", color=color)
        ax.semilogy(x, y_model, linestyle="-", marker="none", color=color, label=f"molality = {molality_PZ}" )
    print(f"Total number of data points: {n_data}")
    print(f"Average relative error in PZ pressure: {P_PZ_rel_err/n_data}")
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("PZ vapor pressure (Pa)")
    ax.set_title("Hillard PZ Volatility")
    ax.legend()

    fig.show()
    
    fig = plt.figure()
    ax = fig.subplots()

    x = []
    y_model = []
    for i in range(1, 100):
        x.append(m.state_test[i].mole_frac_comp["PZ"].value)
        y_model.append(
            pyo.value(m.state_test[i].fug_phase_comp["Liq","PZ"] 
                + m.state_test[i].fug_phase_comp["Liq","H2O"]) / 1e3 # Convert from Pa to kPa
        )
    ax.semilogy(x, y_model)
    fig.show()

    fig = plt.figure()
    ax = fig.subplots()
    ax2 = ax.twinx()
    x = []
    P_H2O_data = []
    P_H2O_model = []
    P_PZ_data = []
    P_PZ_model = []
    print ("\n Data source: Nguyen")
    P_PZ_rel_err = 0
    P_H2O_rel_err = 0
    n_data = 0
    for i, row in df_nguyen_3_3_1.iterrows():
        n_data +=1
        x.append(row["temperature"])
        P_H2O_data.append(row["pressure_H2O"])
        P_H2O_model.append(pyo.value(m.state_nguyen_3_3_1[i].fug_phase_comp["Liq","H2O"]/1e3))
        P_PZ_data.append(row["pressure_PZ"])
        P_PZ_model.append(pyo.value(m.state_nguyen_3_3_1[i].fug_phase_comp["Liq","PZ"]))

    for y_data_pt, y_model_pt in zip(P_H2O_data, P_H2O_model):
        P_H2O_rel_err += abs((y_data_pt-y_model_pt)/y_data_pt) 
    for y_data_pt, y_model_pt in zip(P_PZ_data, P_PZ_model):
        P_PZ_rel_err += abs((y_data_pt-y_model_pt)/y_data_pt) 

    ax.plot(x, P_H2O_data, linestyle="none", marker="o", color="royalblue")
    ax.plot(x, P_H2O_model, linestyle="-", marker="none", color="royalblue", label="P_H2O" )
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("$P_{H2O}$ (kPa)", color="royalblue")
    ax2.plot(x, P_PZ_data, linestyle="none", marker="o", color="firebrick")
    ax2.plot(x, P_PZ_model, linestyle="-", marker="none", color="firebrick", label="P_PZ" )
    ax2.set_ylabel("$P_{PZ}$ (Pa)", color="firebrick")
    ax.set_title("PZ volatility (0.48m)")

    fig.show()

    fig = plt.figure()
    ax = fig.subplots()
    ax2 = ax.twinx()
    x = []
    P_H2O_data = []
    P_H2O_model = []
    P_PZ_data = []
    P_PZ_model = []
    molality=8.0

    for i, row in df_nguyen_5_3_3.iterrows():
        n_data +=1
        x.append(row["temperature"])
        P_H2O_data.append(row["pressure_H2O"])
        P_H2O_model.append(pyo.value(m.state_nguyen_5_3_3[i].fug_phase_comp["Liq","H2O"]/1e3))
        P_PZ_data.append(row["pressure_PZ"])
        P_PZ_model.append(pyo.value(m.state_nguyen_5_3_3[i].fug_phase_comp["Liq","PZ"]))

    for y_data_pt, y_model_pt in zip(P_H2O_data, P_H2O_model):
        P_H2O_rel_err += abs((y_data_pt-y_model_pt)/y_data_pt) 
    for y_data_pt, y_model_pt in zip(P_PZ_data, P_PZ_model):
        P_PZ_rel_err += abs((y_data_pt-y_model_pt)/y_data_pt) 

    print(f"Total number of data points: {n_data}")
    print(f"Average relative error in H2O pressure: {P_H2O_rel_err/n_data}")
    print(f"Average relative error in PZ pressure: {P_PZ_rel_err/n_data}")

    ax.plot(x, P_H2O_data, linestyle="none", marker="o", color="royalblue")
    ax.plot(x, P_H2O_model, linestyle="-", marker="none", color="royalblue", label="P_H2O" )
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("$P_{H2O}$ (kPa)", color="royalblue")
    ax2.plot(x, P_PZ_data, linestyle="none", marker="o", color="firebrick")
    ax2.plot(x, P_PZ_model, linestyle="-", marker="none", color="firebrick", label="P_PZ" )
    ax2.set_ylabel("$P_{PZ}$ (Pa)", color="firebrick")
    ax.set_title("PZ volatility (8.0m)")

    fig.show()

    fig = plt.figure()
    ax = fig.subplots()
    colors = ["firebrick", "royalblue", "forestgreen", "gold", "darkorchid"]

    print ("\n Data source: Chen")
    cp_rel_err = 0
    n_data = 0

    for x_PZ, color in zip(df_chen_54_59["x_PZ"].unique(), colors):
        x = []
        y_data =[]
        y_model = []
        for i, row in df_chen_54_59.loc[df_chen_54_59["x_PZ"] ==  x_PZ].iterrows():
            n_data += 1
            x.append(float(row["temperature"]))
            y_data.append(row["cp"])
            y_model.append(pyo.value(m.state_chen_54_59[i].Liq_cp))
        
        for y_data_pt, y_model_pt in zip(P_PZ_data, P_PZ_model):
            cp_rel_err += abs((y_data_pt-y_model_pt)/y_data_pt) 

        ax.plot(x, y_data, linestyle="none", marker="o", color=color)
        ax.plot(x, y_model, linestyle="-", marker="none", color=color, label=f"mole fraction = {x_PZ}" )


    print(f"Total number of data points: {n_data}")
    print(f"Average relative error in c_p: {cp_rel_err/n_data}")

    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Heat capacity (J/mol K)")
    ax.set_title("")
    ax.legend()

    fig.show()

    print("ok, boomer")