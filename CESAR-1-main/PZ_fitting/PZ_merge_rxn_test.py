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
    return x ** 2

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
    rxn_combinations = {
        "PZ_OH^-_formation": {
            "H2O_autoionization": 1,
            "PZ_protonation": 1
        }
    }
    config = get_prop_dict(["H2O", "PZ"], rxn_combinations=rxn_combinations)

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
        # "H3O^+": 1e11,
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

    for comp, sf_x in mole_frac_true_scaling_factors.items():
        params.set_default_scaling("mole_frac_phase_comp_true", sf_x, index=("Liq", comp))
        params.set_default_scaling(
            "flow_mol_phase_comp_true",
            sf_x * scaling_factor_flow_mol,
            index=("Liq", comp)
        )

    # params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1e4, index="H2O_autoionization")
    # params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1e4, index="PZ_protonation")
    params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1e4, index="PZ_OH^-_formation")

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
    "bound_relax_factor": 0,
    'halt_on_ampl_error': 'yes',
    # 'mu_strategy': 'monotone',
}

if __name__ == "__main__":
    logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)
    init_outlevel = idaeslog.DEBUG
    henry=False

    data = os.sep.join([this_file_dir(), "data"])
    m = pyo.ConcreteModel()
    create_and_scale_params(m, henry=henry)

    m.state_test = m.params.build_state_block(range(1,100), defined_state=True)

    for i in range(1, 100):
        x_PZ = i/100
        m.state_test[i].flow_mol.fix(100)
        m.state_test[i].mole_frac_comp["H2O"].fix(1-x_PZ)
        m.state_test[i].mole_frac_comp["PZ"].fix(x_PZ)
        m.state_test[i].pressure.fix(2e5)
        m.state_test[i].temperature.fix(20 + 273.15)
        # @m.state_test[i].Expression()
        # def pH(b):
        #     return -pyo.log(b.act_phase_comp_true["Liq","H3O^+"]*b.dens_mol_phase["Liq"]*1e-3)/pyo.log(10)
        @m.state_test[i].Expression()
        def pOH(b):
            return -pyo.log(b.act_phase_comp_true["Liq","OH^-"]*b.dens_mol_phase["Liq"]*1e-3)/pyo.log(10)

    iscale.calculate_scaling_factors(m)

    m.state_test.initialize(
        hold_state=False,
        outlvl=init_outlevel,
        optarg=optarg
    )

    m_scaled = pyo.TransformationFactory('core.scale_model').create_using(m, rename=False)
    optarg.pop("nlp_scaling_method", None) # Scaled model doesn't need user scaling
    optarg["max_iter"] = 500
    solver = get_solver(
        "ipopt",
        options=optarg
    )
    # solve_indexed_blocks(solver, m.state_chen_54_59, tee=True)
    res = solver.solve(m_scaled, tee=True)
    pyo.assert_optimal_termination(res)
    
    fig = plt.figure()
    ax = fig.subplots()
    ax2 = ax.twinx()

    x = []
    # pH = []
    pOH = []
    for i in range(1, 100):
        x.append(m.state_test[i].mole_frac_comp["PZ"].value)
        # pH.append(
        #     pyo.value(m.state_test[i].pH)
        # )
        pOH.append(
            pyo.value(m.state_test[i].pOH)
        )
    # ax.plot(x, pH, color="red")
    ax2.plot(x, pOH, color="blue")
    ax.set_xlabel("PZ mole frac")
    ax.set_ylabel("pH")
    ax2.set_ylabel("pOH")
    fig.show()

    print("ok, boomer")