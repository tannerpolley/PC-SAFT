import sys, os, logging
import time

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
from idaes.core.util.math import smooth_abs
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale
from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
from idaes.models.properties.modular_properties.pure import NIST
from CESAR_1_eNRTL import get_prop_dict, create_heat_capacity_no_inherent_rxns, initialize_inherent_reactions



def loss_least_squares(x):
    return 0.5 * x ** 2


def loss_abs(x):
    return smooth_abs(x, 1e-2)


def get_estimated_params(m):
    param_list = []
    for rxn_name in [
        # PZ bicarbonate parameters totally characterized from pKa data
        # of PZ and CO2 in H2O
        # "PZ_bicarbonate_formation",
        "PZ_carbamate_formation_combo",
        "PZ_carbamate_proton_transfer",
        "PZ_bicarbamate_formation_combo",
    ]:
        rxn_obj = getattr(m.params, "reaction_" + rxn_name)
        param_list.append(rxn_obj.k_eq_coeff_1)
        param_list.append(rxn_obj.k_eq_coeff_2)
        # param_list.append(rxn_obj.k_eq_coeff_3)

    for idx in [
        "PZH^+, PZCOO^-",
        "PZH^+, PZ(COO^-)_2",
        "PZH^+, HCO3^-",
        "PZH^+COO^-",
    ]:
        param_list.append(m.params.Liq.tau_A["H2O", idx])
        param_list.append(m.params.Liq.tau_B["H2O", idx])
        # param_list.append(m.params.Liq.tau_A["PZ", idx])
        # param_list.append(m.params.Liq.tau_B["PZ", idx])
        # param_list.append(m.params.Liq.tau_A[idx, "H2O"])
        # param_list.append(m.params.Liq.tau_B[idx, "H2O"])
        # param_list.append(m.params.Liq.tau_A["CO2", idx])
        # param_list.append(m.params.Liq.tau_B["CO2", idx])

    # param_list.append(m.params.Liq.tau_A["H2O", "PZ"])
    # param_list.append(m.params.Liq.tau_B["H2O", "PZ"])
    # param_list.append(m.params.Liq.tau_A["PZ", "H2O"])
    # param_list.append(m.params.Liq.tau_B["PZ", "H2O"])
    # param_list.append(m.params.Liq.alpha["H2O","PZ"])
    return param_list


def create_and_scale_params(m):
    rxn_combinations = {
        # "PZ_OH^-_formation": {
        #     "H2O_autoionization": 1,
        #     "PZ_protonation": 1
        # }
        "PZ_bicarbonate_formation": {
            "bicarbonate_formation": 1,
            "PZ_protonation": 1
        },
        "PZ_carbamate_formation_combo": {
            "PZ_carbamate_formation": 1,
            "PZ_protonation": 1
        },
        "PZ_carbamate_proton_transfer": {
            "PZ_carbamate_protonation": 1,
            "PZ_protonation": -1
        },
        "PZ_bicarbamate_formation_combo": {
            "PZ_bicarbamate_formation": 1,
            "PZ_protonation": 1
        }
    }
    config = get_prop_dict(["H2O", "PZ", "CO2"],
                           excluded_rxns=["H2O_autoionization", "carbonate_formation"],
                           rxn_combinations=rxn_combinations
                           )
    assert "H3O^+" not in config["components"]
    assert "OH^-" not in config["components"]
    params = m.params = GenericParameterBlock(**config)

    gsf = iscale.get_scaling_factor
    scaling_factor_flow_mol = 1 / 100
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
        "CO2": 5
    }
    mole_frac_true_scaling_factors = {
        "PZ": 1e1,
        "H2O": 2,
        "HCO3^-": 5e3,
        # "H3O^+": 5e5,
        # "CO3^2-": 1e12,
        # "OH^-": 5e11,
        # "HCO3^-": 1e3,
        # "H3O^+": 1e11,
        # "CO3^2-": 1e3,
        # "OH^-": 1e3,
        "PZH^+": 1e1,
        "PZCOO^-": 5e2,
        "PZH^+COO^-": 5e2,
        "PZ(COO^-)_2": 5e3,
        "CO2": 1e3

    }
    inherent_rxn_scaling_factors = {
        "PZ_bicarbonate_formation": 5e3,
        "PZ_carbamate_formation_combo": 5e2,
        "PZ_carbamate_proton_transfer": 5e2,
        "PZ_bicarbamate_formation_combo": 5e3,
    }
    for comp, sf_x in mole_frac_scaling_factors.items():
        params.set_default_scaling("mole_frac_comp", sf_x, index=comp)
        params.set_default_scaling("mole_frac_phase_comp", sf_x, index=("Liq", comp))
        params.set_default_scaling("flow_mol_phase_comp", sf_x * scaling_factor_flow_mol, index=("Liq", comp))

    for comp, sf_x in mole_frac_true_scaling_factors.items():
        params.set_default_scaling("mole_frac_phase_comp_true", sf_x, index=("Liq", comp))
        params.set_default_scaling("flow_mol_phase_comp_true", sf_x * scaling_factor_flow_mol, index=("Liq", comp))

    for rxn, sf_xi in inherent_rxn_scaling_factors.items():
        params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol * sf_xi, index=rxn)

    iscale.set_scaling_factor(m.params.Liq.alpha, 1)
    iscale.set_scaling_factor(m.params.Liq.tau_A, 1)  # Reminder that it's well-scaled
    iscale.set_scaling_factor(m.params.Liq.tau_B, 1 / 300)

    for rxn_name in inherent_rxn_scaling_factors.keys():
        rxn_obj = getattr(m.params, "reaction_" + rxn_name)
        iscale.set_scaling_factor(rxn_obj.k_eq_coeff_1, 1)
        iscale.set_scaling_factor(rxn_obj.k_eq_coeff_2, 1 / 300)
        iscale.set_scaling_factor(rxn_obj.k_eq_coeff_3, 1)
        iscale.set_scaling_factor(rxn_obj.k_eq_coeff_4, 300)


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
    init_outlevel = idaeslog.INFO_LOW

    t0 = time.time()

    data = os.sep.join([this_file_dir(), "data"])
    m = pyo.ConcreteModel()
    create_and_scale_params(m)
    fit_dH_abs = True
    plot_dH_abs = True
    fit_speciation = False

    # m.params.Liq.tau_A["H2O", "PZ"].set_value(5.32)
    # m.params.Liq.tau_B["H2O", "PZ"].set_value(-1279.99)
    # m.params.Liq.tau_A["PZ", "H2O"].set_value(-0.6)
    # m.params.Liq.tau_B["PZ", "H2O"].set_value(-698.51)
    # m.params.Liq.alpha["H2O", "PZ"].set_value(0.3)

    obj_least_squares_expr = 0
    obj_abs_expr = 0

    df_hillard_loading = pd.read_csv(os.sep.join([data, "hillard_PZ_loading_concatenated.csv"]), index_col=None)

    n_data_hillard_loading = len(df_hillard_loading["temperature"])
    m.hillard_loading = m.params.build_state_block(range(n_data_hillard_loading), defined_state=True)

    for i, row in df_hillard_loading.iterrows():
        molality = row["PZ_molality"]
        n_PZ = molality
        n_H2O = 1 / 0.01802
        # Convention in papers from Gary Rochelle's group to assume 1 mo le PZ can dissolve 2 moles CO2
        n_CO2 = 2 * n_PZ * row["CO2_loading"]
        n_tot = n_PZ + n_H2O + n_CO2

        m.hillard_loading[i].flow_mol.fix(100)
        m.hillard_loading[i].mole_frac_comp["H2O"].fix(n_H2O / n_tot)
        m.hillard_loading[i].mole_frac_comp["PZ"].fix(n_PZ / n_tot)
        m.hillard_loading[i].mole_frac_comp["CO2"].fix(n_CO2 / n_tot)

        # Got 45 psig as the average temperature from Cuillane and Rochelle (2005)
        # Assuming it's the same here, because I can't find Hillard's master thesis.
        # Total pressure doesn't matter at this stage
        m.hillard_loading[i].pressure.fix(310264 + 101300)
        m.hillard_loading[i].temperature.fix(row["temperature"] + 273.15)  # Temperature in C

        # CO2 partial pressure
        obj_least_squares_expr += (loss_least_squares(m.hillard_loading[i].log_fug_phase_comp["Liq", "CO2"]
                                                      - pyo.log(row["CO2_partial_pressure"] * 1e3)))  # Pressure in kPa
        # CO2 partial pressure
        obj_abs_expr += (loss_abs(m.hillard_loading[i].log_fug_phase_comp["Liq", "CO2"]
                - pyo.log(row["CO2_partial_pressure"] * 1e3)))  # Pressure in kPa


    df_dugas_loading = pd.read_csv(os.sep.join([data, "dugas_table_4_5.csv"]), index_col=None)

    n_data_dugas_loading = len(df_dugas_loading["temperature"])
    m.dugas_loading = m.params.build_state_block(range(n_data_dugas_loading), defined_state=True)

    for i, row in df_dugas_loading.iterrows():
        molality = row["PZ_molality"]
        n_PZ = molality
        n_H2O = 1 / 0.01802
        # Convention in papers from Gary Rochelle's group to assume 1 mole PZ can dissolve 2 moles CO2
        n_CO2 = 2 * n_PZ * row["CO2_loading"]
        n_tot = n_PZ + n_H2O + n_CO2

        m.dugas_loading[i].flow_mol.fix(100)
        m.dugas_loading[i].mole_frac_comp["H2O"].fix(n_H2O / n_tot)
        m.dugas_loading[i].mole_frac_comp["PZ"].fix(n_PZ / n_tot)
        m.dugas_loading[i].mole_frac_comp["CO2"].fix(n_CO2 / n_tot)
        # Using 45 psig as the average pressure 
        # Actual pressure data is in the appendix, but total pressure doesn't
        # affect anything at the moment
        m.dugas_loading[i].pressure.fix(310264 + 101300)
        m.dugas_loading[i].temperature.fix(row["temperature"] + 273.15)  # Temperature in C
        obj_least_squares_expr += (
            # CO2 partial pressure
            loss_least_squares(
                m.dugas_loading[i].log_fug_phase_comp["Liq", "CO2"]
                - pyo.log(row["CO2_equilibrium_partial_pressure"])  # Pressure in Pa
            )
        )
        obj_abs_expr += (
            # CO2 partial pressure
            loss_abs(
                m.dugas_loading[i].log_fug_phase_comp["Liq", "CO2"]
                - pyo.log(row["CO2_equilibrium_partial_pressure"])  # Pressure in Pa
            )
        )

    if fit_dH_abs or plot_dH_abs:
        df_hartono_dH_abs = pd.read_csv(os.sep.join([data, "hartono_table_15.csv"]), index_col=None)
        n_data_hartono_dH_abs = len(df_hartono_dH_abs)
        # Parse dataframe into dictionaries
        dict_hartono_dH_abs = {}
        n_test = 0
        for T in df_hartono_dH_abs["temperature"].unique():
            n_test += 1
            dict_hartono_dH_abs[T] = {
                "loading": np.array(
                    df_hartono_dH_abs.loc[df_hartono_dH_abs["temperature"] == T]["loading"]
                ),
                "loading_uncertainty": np.array(
                    df_hartono_dH_abs.loc[df_hartono_dH_abs["temperature"] == T]["loading_uncertainty"]
                ),
                "dH_abs": np.array(
                    df_hartono_dH_abs.loc[df_hartono_dH_abs["temperature"] == T]["dH_abs"]
                ),
                "dH_abs_uncertainty": np.array(
                    df_hartono_dH_abs.loc[df_hartono_dH_abs["temperature"] == T]["dH_uncertainty"]
                ),
            }
        print(dict_hartono_dH_abs)
        m.hartono_dH_blk = m.params.build_state_block(range(n_data_hartono_dH_abs + n_test), defined_state=True)
        idx_start = 0
        for T, subdict in dict_hartono_dH_abs.items():
            subdict["range"] = range(idx_start, idx_start + len(subdict["loading"] + 1))
            # print(subdict["range"])
            # Start out with a completely unloaded mixture
            molality_PZ = 2.1
            n_H2O = pyo.value(1 / m.params.H2O.mw)
            n_CO2 = 0.01
            n_tot = molality_PZ + n_H2O + n_CO2
            blk = m.hartono_dH_blk[idx_start]
            blk.flow_mol.fix(n_tot)
            blk.mole_frac_comp["H2O"].fix(n_H2O / n_tot)
            blk.mole_frac_comp["PZ"].fix(molality_PZ / n_tot)
            blk.mole_frac_comp["CO2"].fix(n_CO2 / n_tot)
            # Not a lot of information about pressure
            blk.pressure.fix(2e5)
            blk.temperature.fix(T)
            enth_abs_list = []
            enth_component_dict = {
                "phys": [],
                "excess": [],
                "PZ_bicarbonate_formation": [],
                "PZ_carbamate_formation_combo": [],
                "PZ_carbamate_proton_transfer": [],
                "PZ_bicarbamate_formation_combo": [],
            }
            for k, loading in enumerate(subdict["loading"]):
                blk_old = blk
                blk = m.hartono_dH_blk[idx_start + k + 1]
                n_CO2 = molality_PZ * loading
                n_tot = n_PZ + n_H2O + n_CO2
                blk.flow_mol.fix(n_tot)
                blk.mole_frac_comp["H2O"].fix(n_H2O / n_tot)
                blk.mole_frac_comp["PZ"].fix(molality_PZ / n_tot)
                blk.mole_frac_comp["CO2"].fix(n_CO2 / n_tot)
                # Not a lot of information about pressure
                blk.pressure.fix(2e5)
                blk.temperature.fix(T)
                CO2_obj = m.params.CO2
                print(blk.flow_mol.value - blk_old.flow_mol.value)
                dH_abs_expr = -(  # Negative sign bc values reported are positive
                        blk.energy_internal_mol_phase["Liq"] * blk.flow_mol
                        - blk_old.energy_internal_mol_phase["Liq"] * blk_old.flow_mol
                        - NIST.enth_mol_ig_comp.return_expression(
                    blk,
                    CO2_obj,
                    blk.temperature
                ) * (blk.flow_mol_comp["CO2"] - blk_old.flow_mol_comp["CO2"])
                    # There should be an additional term for PV work to pressurize
                    # the vapor phase with CO2, but we can't account for it because
                    # we have limited info about the vapor phase and the pressurized
                    # cylinder the CO2 was being discharged from
                ) / (blk.flow_mol_comp["CO2"] - blk_old.flow_mol_comp["CO2"])
                enth_abs_list.append(dH_abs_expr)

                enth_component_dict["phys"].append(
                    -8.314462618 * pyo.units.joule / pyo.units.mol / pyo.units.degK * (
                            CO2_obj.henry_coeff_1
                            - CO2_obj.henry_coeff_2 * T
                            - CO2_obj.henry_coeff_3 * T ** 2
                    )
                )
                enth_component_dict["excess"].append(
                    -(
                            blk.enth_mol_phase_excess["Liq"]
                            * sum(
                        blk.flow_mol_phase_comp_true["Liq", j]
                        for j in blk.components_in_phase("Liq")
                    )
                            - blk_old.enth_mol_phase_excess["Liq"]
                            * sum(
                        blk.flow_mol_phase_comp_true["Liq", j]
                        for j in blk.components_in_phase("Liq")
                    )
                    ) / (blk.flow_mol_comp["CO2"] - blk_old.flow_mol_comp["CO2"])
                )
                for rxn in [
                    "PZ_bicarbonate_formation",
                    "PZ_carbamate_formation_combo",
                    "PZ_carbamate_proton_transfer",
                    "PZ_bicarbamate_formation_combo",
                ]:
                    enth_component_dict[rxn].append(-(blk.dh_rxn[rxn] * blk.apparent_inherent_reaction_extent[rxn]
                                                    - blk_old.dh_rxn[rxn] * blk_old.apparent_inherent_reaction_extent[rxn])
                                                    / (blk.flow_mol_comp["CO2"] - blk_old.flow_mol_comp["CO2"]))

                if fit_dH_abs and (loading <= 0.75):  # threshold bc of missing vapor phase enthalpy
                    residual = ((dH_abs_expr * 1e-3 - subdict["dH_abs"][k]) / subdict["dH_abs_uncertainty"][k])
                    obj_least_squares_expr += loss_least_squares(residual)
                    obj_abs_expr += loss_abs(residual)

            subdict["enth_abs_expr"] = enth_abs_list
            subdict["enth_component_dict"] = enth_component_dict
            idx_start += len(subdict["loading"]) + 1

    #################################################################################
    ##### Speciation Data
    #################################################################################
    if fit_speciation:
        df_ermatchkov_speciation = pd.read_csv(os.sep.join([data, "ermatchkov_et_al_2003.csv"]), index_col=None)

        n_data_ermatchkov_speciation = len(df_ermatchkov_speciation["temperature"])
        # TODO need to correct for deuterated solvent
        m.ermatchkov_speciation = m.params.build_state_block(
            range(n_data_ermatchkov_speciation),
            defined_state=True
        )

        for i, row in df_ermatchkov_speciation.iterrows():
            n_PZ = row["moles_PZ_dissolved"]
            n_H2O = 1 / 0.02003
            n_CO2 = row["moles_CO2_dissolved"]
            n_tot = n_PZ + n_H2O + n_CO2

            m.ermatchkov_speciation[i].flow_mol.fix(n_tot)
            m.ermatchkov_speciation[i].mole_frac_comp["H2O"].fix(n_H2O / n_tot)
            m.ermatchkov_speciation[i].mole_frac_comp["PZ"].fix(n_PZ / n_tot)
            m.ermatchkov_speciation[i].mole_frac_comp["CO2"].fix(n_CO2 / n_tot)
            m.ermatchkov_speciation[i].pressure.fix(1e5)
            m.ermatchkov_speciation[i].temperature.fix(row["temperature"])  # Temperature in K
            moles_PZ_plus_deuterons = (
                    m.ermatchkov_speciation[i].flow_mol_phase_comp_true["Liq", "PZ"]
                    + m.ermatchkov_speciation[i].flow_mol_phase_comp_true["Liq", "PZH^+"]
                # Doubly deuterated PZ would go here (PZ(H^+)_2), but it's not in our model
            )
            moles_monocarbamate = (
                    m.ermatchkov_speciation[i].flow_mol_phase_comp_true["Liq", "PZCOO^-"]
                    + m.ermatchkov_speciation[i].flow_mol_phase_comp_true["Liq", "PZH^+COO^-"]
            )
            moles_bicarbamate = m.ermatchkov_speciation[i].flow_mol_phase_comp_true["Liq", "PZ(COO^-)_2"]
            if not np.isnan(row["moles_PZ_plus_deuterons"]):
                residual = moles_PZ_plus_deuterons - row["moles_PZ_plus_deuterons"]
                obj_least_squares_expr += loss_least_squares(residual)
                obj_abs_expr += loss_abs(residual)
            if not np.isnan(row["moles_monocarbamate"]):
                residual = moles_monocarbamate - row["moles_monocarbamate"]
                obj_least_squares_expr += loss_least_squares(residual)
                obj_abs_expr += loss_abs(residual)
            if not np.isnan(row["moles_bicarbamate"]):
                residual = moles_bicarbamate - row["moles_bicarbamate"]
                obj_least_squares_expr += loss_least_squares(residual)
                obj_abs_expr += loss_abs(residual)

    iscale.calculate_scaling_factors(m)
    print(f"DOF: {degrees_of_freedom(m)}")
    initialize_inherent_reactions(m.hillard_loading)
    m.hillard_loading.initialize(
        hold_state=False,
        outlvl=init_outlevel,
        optarg=optarg
    )

    initialize_inherent_reactions(m.dugas_loading)
    m.dugas_loading.initialize(
        hold_state=False,
        outlvl=init_outlevel,
        optarg=optarg
    )
    if fit_dH_abs or plot_dH_abs:
        initialize_inherent_reactions(m.hartono_dH_blk)
        m.hartono_dH_blk.initialize(
            hold_state=False,
            outlvl=init_outlevel,
            optarg=optarg
        )
    if fit_speciation:
        initialize_inherent_reactions(m.ermatchkov_speciation)
        m.ermatchkov_speciation.initialize(
            hold_state=False,
            outlvl=init_outlevel,
            optarg=optarg
        )
    # Need to create heat capacity equations after initialization because
    # initialization method doesn't know about them
    # create_heat_capacity_no_inherent_rxns(m.state_chen_54_59)
    # assert degrees_of_freedom(m.state_chen_54_59) == 0
    # for i, row in df_chen_54_59.iterrows():
    #     obj_expr +=  (
    #         # H2O partial pressure
    #         loss(
    #             0.1*(m.state_chen_54_59[i].Liq_cp - float(row["cp"]))
    #         )
    #     )

    iscale.calculate_scaling_factors(m)

    estimated_vars = get_estimated_params(m)
    # Apparently the model reduces to the Margules model at alpha=0,
    # which corresponds to random mixing. Alpha values less than zero
    # are not thermodynamically meaningful, because you can't get more
    # random than fully random.
    # m.params.Liq.alpha["H2O", "PZ"].lb = 0
    # m.params.Liq.alpha["H2O", "PZ"].ub = 1

    for var in estimated_vars:
        obj_least_squares_expr += 0.01 * loss_least_squares(
            (var - var.value) * iscale.get_scaling_factor(var)
        )
        obj_abs_expr += 0.01 * loss_abs(
            (var - var.value) * iscale.get_scaling_factor(var)
        )
    print("Creating Objective Function")
    m.obj_least_squares = pyo.Objective(expr=obj_least_squares_expr)
    m.obj_abs = pyo.Objective(expr=obj_abs_expr)
    m.obj_abs.deactivate()
    print("Scaling Transformation")
    m_scaled = pyo.TransformationFactory('core.scale_model').create_using(m, rename=False)
    optarg.pop("nlp_scaling_method", None)  # Scaled model doesn't need user scaling
    optarg["max_iter"] = 500
    solver = get_solver(
        "ipopt",
        options=optarg
    )

    estimated_vars_scaled = get_estimated_params(m_scaled)
    for var in estimated_vars_scaled:
        var.unfix()
    print("Solving for least squares")
    res = solver.solve(m_scaled, tee=True)
    pyo.assert_optimal_termination(res)


    pyo.TransformationFactory('core.scale_model').propagate_solution(m_scaled, m)
    print("==================================================")
    print("========== Variables with uncertainty ============")
    print("==================================================")


    def gsf(var):
        return iscale.get_scaling_factor(var, default=1)


    # for i, var in enumerate(estimated_vars):
    #     print(f"{var.name}: {var.value:.3e} +/- {pyo.sqrt(inv_red_hess[1][i][i])/gsf(var):.3e}")

    for i, var in enumerate(estimated_vars):
        print(f"{var.name}: {var.value:.3e}")

    colors = [
        "firebrick",
        "royalblue",
        "forestgreen",
        "goldenrod",
        "magenta",
        "orangered",
        "cyan",
        "indigo",
    ]
    print("\n Data source: Hillard")
    P_CO2_rel_err = 0
    n_data = 0
    for molality in df_hillard_loading["PZ_molality"].unique():
        fig = plt.figure()
        ax = fig.subplots()
        df_hillard_molality = df_hillard_loading.loc[df_hillard_loading["PZ_molality"] == molality]
        for j, T in enumerate(df_hillard_molality["temperature"].unique()):
            x = []
            y_data = []
            y_model = []
            for i, row in df_hillard_molality.loc[df_hillard_molality["temperature"] == T].iterrows():
                n_data += 1
                loading = float(row["CO2_loading"])
                x.append(loading)
                y_data.append(row["CO2_partial_pressure"])
                candidate_idx = np.where(
                    df_hillard_loading["CO2_partial_pressure"].values == row["CO2_partial_pressure"])
                assert len(candidate_idx) == 1
                assert len(candidate_idx[0]) == 1
                y_model.append(
                    pyo.value(m.hillard_loading[candidate_idx[0][0]].fug_phase_comp["Liq", "CO2"] / 1e3)
                    # Convert from Pa to kPa
                )

            for y_data_pt, y_model_pt in zip(y_data, y_model):
                P_CO2_rel_err += abs((y_data_pt - y_model_pt) / y_data_pt)

            ax.semilogy(x, y_data, linestyle="none", marker="o", color=colors[j])
            ax.semilogy(x, y_model, linestyle="-", color=colors[j], label=f"T = {T}")

        ax.set_xlabel("CO2 Loading (Rochelle convention)")
        ax.set_ylabel("CO2 vapor pressure (kPa)")
        ax.set_title(f"Hillard Loading PZ molality = {molality}")
        ax.legend()
        fig.show()
    print(f"Total number of data points: {n_data}")
    print(f"Average relative error in Co2 pressure: {P_CO2_rel_err / n_data}")

    print("\n Data source: Dugas")
    P_CO2_rel_err = 0
    n_data = 0

    for molality in df_dugas_loading["PZ_molality"].unique():
        fig = plt.figure()
        ax = fig.subplots()
        df_dugas_molality = df_dugas_loading.loc[df_dugas_loading["PZ_molality"] == molality]
        for j, T in enumerate(df_dugas_molality["temperature"].unique()):
            x = []
            y_data = []
            y_model = []
            for i, row in df_dugas_molality.loc[df_dugas_molality["temperature"] == T].iterrows():
                n_data += 1
                loading = float(row["CO2_loading"])
                x.append(loading)
                y_data.append(row["CO2_equilibrium_partial_pressure"])
                candidate_idx = np.where(df_dugas_loading["CO2_equilibrium_partial_pressure"].values == row[
                    "CO2_equilibrium_partial_pressure"])
                assert len(candidate_idx) == 1
                assert len(candidate_idx[0]) == 1
                y_model.append(
                    pyo.value(m.dugas_loading[candidate_idx[0][0]].fug_phase_comp["Liq", "CO2"])
                    # Convert from Pa to kPa
                )

            for y_data_pt, y_model_pt in zip(y_data, y_model):
                P_CO2_rel_err += abs((y_data_pt - y_model_pt) / y_data_pt)

            ax.semilogy(x, y_data, linestyle="none", marker="o", color=colors[j])
            ax.semilogy(x, y_model, linestyle="-", color=colors[j], label=f"T = {T}")
        ax.set_xlabel("CO2 Loading (Rochelle convention)")
        ax.set_ylabel("CO2 vapor pressure (Pa)")
        ax.set_title(f"Dugas Loading PZ molality = {molality}")
        ax.legend()
        fig.show()

    print(f"Total number of data points: {n_data}")
    print(f"Average relative error in CO2 pressure: {P_CO2_rel_err / n_data}")

    enth_comps = [
        "phys",
        "excess",
        "PZ_bicarbonate_formation",
        "PZ_carbamate_formation_combo",
        "PZ_carbamate_proton_transfer",
        "PZ_bicarbamate_formation_combo",
    ]


    def plot_enth_components(x, enth_dict, T):
        fig = plt.figure()
        ax = fig.subplots()

        # From StackExchange
        data = np.array([array for array in enth_dict.values()])
        data_shape = np.shape(data)

        def get_cumulated_array(data, **kwargs):
            cum = data.clip(**kwargs)
            cum = np.cumsum(cum, axis=0)
            d = np.zeros(np.shape(data))
            return d

        cumulated_data = get_cumulated_array(data, min=0)
        cumulated_data_neg = get_cumulated_array(data, max=0)

        row_mask = (data < 0)
        cumulated_data[row_mask] = cumulated_data_neg[row_mask]
        data_stack = cumulated_data

        colors = [
            "firebrick",
            "royalblue",
            "forestgreen",
            "goldenrod",
            "magenta",
            "orangered"
        ]
        labels = [key for key in enth_dict.keys()]
        for i in np.arange(0, data_shape[0]):
            ax.bar(
                np.arange(data_shape[1]),
                # x,
                data[i],
                bottom=data_stack[i],
                color=colors[i],
                label=labels[i]
            )
        ax.legend()
        ax.set_xlabel("CO2 Loading")
        ax.set_ylabel("Enthalpy (kJ/mol)")
        ax.set_title(f"Components of Enthalpy of Absorption T={T}")
        fig.show()


    if plot_dH_abs:
        fig = plt.figure()
        ax = fig.subplots()

        for i, T in enumerate(dict_hartono_dH_abs.keys()):
            subdict = dict_hartono_dH_abs[T]
            x = []
            y_data = []
            y_model = []
            arrays = {}
            for comp in enth_comps:
                arrays[comp] = []
            for k, loading in enumerate(subdict["loading"]):
                x.append(loading)
                y_data.append(subdict["dH_abs"][k])
                y_model.append(1e-3 * pyo.value(subdict["enth_abs_expr"][k]))
                for comp in enth_comps:
                    arrays[comp].append(
                        pyo.value(
                            subdict["enth_component_dict"][comp][k]
                        ) * 1e-3
                    )
            plot_enth_components(x, arrays, T)

            ax.plot(x, y_data, linestyle="none", marker="o", color=colors[i])
            ax.plot(x, y_model, linestyle="-", color=colors[i], label=f"T = {T}")
        ax.plot(
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            1e-3 * np.array([75227.28143, 73122.76415, 71018.24687, 68913.72959, 66809.21231, 64704.69503]),
            linestyle="-",
            color=colors[4],
            label="Hillard Gibbs-Helmholtz 323.15K"
        )
        ax.plot(
            [0.48, 0.632, 0.704, 0.822],
            1e-3 * np.array([76369.54152, 70556.89539, 63084.05525, 55050.88927]),
            linestyle="none",
            marker="x",
            color=colors[5],
            label="Dugas Gibbs-Helmholtz 323.15K"
        )
        ax.plot(
            [0.48, 0.64],
            1e-3 * np.array([73067.05344, 76933.45639]),
            linestyle="none",
            marker="x",
            color=colors[6],
            label="Dugas Gibbs-Helmholtz 343.15K"
        )
        ax.plot(
            [0.48, 0.64],
            1e-3 * np.array([73668.02424, 63762.15438]),
            linestyle="none",
            marker="x",
            color=colors[7],
            label="Dugas Gibbs-Helmholtz 363.15K"
        )
        ax.set_xlabel("CO2 Loading")
        ax.set_ylabel("Enth_Abs (kJ/mol)")
        ax.set_title("Hartono et al. Enthalpy of Absorption")
        ax.legend()
        fig.show()

    if fit_speciation:
        print("\n Data source: Ermatchkov")
        n_PZ_rel_err = 0
        n_carb_rel_err = 0
        n_bicarb_rel_err = 0
        n_data_PZ = 0
        n_data_carb = 0
        n_data_bicarb = 0

        for T in df_ermatchkov_speciation["temperature"].unique():
            df_ermatchkov_temperature = df_ermatchkov_speciation.loc[
                df_ermatchkov_speciation["temperature"] == T
                ]
            for j, nom_molality in enumerate(df_ermatchkov_temperature["nominal_molality_PZ"].unique()):
                x = []
                y_PZ = []
                y_PZ_model = []
                y_carb = []
                y_carb_model = []
                y_bicarb = []
                y_bicarb_model = []
                for i, row in df_ermatchkov_temperature.loc[
                    df_ermatchkov_temperature["nominal_molality_PZ"] == nom_molality].iterrows():
                    n_CO2 = float(row["moles_CO2_dissolved"])
                    n_PZ = float(row["moles_PZ_dissolved"])
                    x.append(n_CO2 / n_PZ)

                    y_PZ.append(row["moles_PZ_plus_deuterons"])
                    y_carb.append(row["moles_monocarbamate"])
                    y_bicarb.append(row["moles_bicarbamate"])

                    candidate_idx = np.where(
                        (df_ermatchkov_speciation["moles_PZ_plus_deuterons"].values == row["moles_PZ_plus_deuterons"])
                        * (df_ermatchkov_speciation["moles_monocarbamate"].values == row["moles_monocarbamate"])
                    )
                    assert len(candidate_idx) == 1
                    assert len(candidate_idx[0]) == 1
                    idx = candidate_idx[0][0]
                    y_PZ_model.append(pyo.value(
                        m.ermatchkov_speciation[idx].flow_mol_phase_comp_true["Liq", "PZ"]
                        + m.ermatchkov_speciation[idx].flow_mol_phase_comp_true["Liq", "PZH^+"]
                        # Doubly deuterated PZ would go here (PZ(H^+)_2), but it's not in our model
                    ))
                    y_carb_model.append(pyo.value(
                        m.ermatchkov_speciation[idx].flow_mol_phase_comp_true["Liq", "PZCOO^-"]
                        + m.ermatchkov_speciation[idx].flow_mol_phase_comp_true["Liq", "PZH^+COO^-"]
                    ))
                    y_bicarb_model.append(
                        pyo.value(m.ermatchkov_speciation[idx].flow_mol_phase_comp_true["Liq", "PZ(COO^-)_2"])
                    )
                n_nan = 0
                for y_data_pt, y_model_pt in zip(y_PZ, y_PZ_model):
                    if not np.isnan(y_data_pt):
                        n_PZ_rel_err += abs((y_data_pt - y_model_pt) / y_data_pt)
                    else:
                        n_nan += 1
                n_data_PZ += len(y_PZ) - n_nan
                n_nan = 0
                for y_data_pt, y_model_pt in zip(y_carb, y_carb_model):
                    if not np.isnan(y_data_pt):
                        n_carb_rel_err += abs((y_data_pt - y_model_pt) / y_data_pt)
                    else:
                        n_nan += 1
                n_data_carb += len(y_carb) - n_nan
                n_nan = 0
                for y_data_pt, y_model_pt in zip(y_bicarb, y_bicarb_model):
                    if not np.isnan(y_data_pt):
                        n_bicarb_rel_err += abs((y_data_pt - y_model_pt) / y_data_pt)
                    else:
                        n_nan += 1
                n_data_bicarb += len(y_bicarb) - n_nan

                fig = plt.figure()
                ax = fig.subplots()

                ax.plot(x, y_PZ, linestyle="none", marker="o", color=colors[0])
                ax.plot(x, y_PZ_model, linestyle="-", color=colors[0], label=f"PZ, PZD^+, PZD^2+")
                ax.plot(x, y_carb, linestyle="none", marker="o", color=colors[1])
                ax.plot(x, y_carb_model, linestyle="-", color=colors[1], label=f"PZCOO^-, PZD^+COO^-")
                ax.plot(x, y_bicarb, linestyle="none", marker="o", color=colors[2])
                ax.plot(x, y_bicarb_model, linestyle="-", color=colors[2], label=f"PZ(COO^-)2")

                ax.set_xlabel("CO2 Loading")
                ax.set_ylabel("Moles species")
                ax.set_title(f"Ermatchkov Speciation PZ T = {T} molal PZ= {nom_molality}")
                ax.legend()
                fig.show()

        print(f"Total number of data points: {n_data}")
        print(f"Average relative error in PZ speciation: {n_PZ_rel_err / n_data_PZ}")
        print(f"Average relative error in monocarbamate speciation: {n_carb_rel_err / n_data_carb}")
        print(f"Average relative error in bicarbamate speciation: {n_bicarb_rel_err / n_data_bicarb}")

    print(f"Time: {time.time() - t0}")
