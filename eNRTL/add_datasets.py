import numpy as np
import pyomo.environ as pyo
from idaes.models.properties.modular_properties.pure import NIST


def loss(x):
    return 0.5 * x ** 2


def get_x(CO2_loading, w_MEA):
    MW_MEA = 61.084
    MW_H2O = 18.02

    x_MEA_unloaded = w_MEA / (MW_MEA / MW_H2O + w_MEA * (1 - MW_MEA / MW_H2O))
    x_H2O_unloaded = 1 - x_MEA_unloaded

    n_MEA = 100 * x_MEA_unloaded
    n_H2O = 100 * x_H2O_unloaded

    n_CO2 = n_MEA * CO2_loading
    n_tot = n_MEA + n_H2O + n_CO2
    x_CO2, x_MEA, x_H2O = n_CO2 / n_tot, n_MEA / n_tot, n_H2O / n_tot

    return np.float32(x_CO2), np.float32(x_MEA), np.float32(x_H2O), n_tot


def add_VLE_dataset(params, df, obj_expr, has_total_pressure=True):

    for i, row in df.iterrows():
        blk = params[i]
        w_MEA = row['MEA_weight_fraction']
        CO2_loading = row['CO2_loading']

        x_CO2, x_MEA, x_H2O, n_tot = get_x(CO2_loading, w_MEA)
        blk.flow_mol.fix(n_tot)
        blk.mole_frac_comp["H2O"].fix(x_H2O)
        blk.mole_frac_comp["MEA"].fix(x_MEA)
        blk.mole_frac_comp["CO2"].fix(x_CO2)
        if has_total_pressure:
            blk.pressure.fix(row['total_pressure'] * 1e3)  # Pressure in kPa
        # else:
        #     blk.pressure.fix(300*1e3)
        blk.temperature.fix(row['temperature'] + 273.15)  # Temperature in C
        logfug_CO2 = blk.log_fug_phase_comp["Liq", "CO2"]
        log_P_CO2_data = pyo.log(row['CO2_pressure'] * 1e3)
        if .1 < CO2_loading < .6:
            obj_expr += loss(logfug_CO2 - log_P_CO2_data)  # Pressure in kPa))

    return obj_expr


def add_ABS_dataset(m, df, params, obj_expr):

    dict_kim_dH_abs = {}
    for T in df["temperature"].unique():
        T = float(T)
        dict_kim_dH_abs[T] = {
            "CO2_loading": np.array(
                df.loc[df["temperature"] == T]["CO2_loading"]
            ),
            "dH_abs": np.array(
                df.loc[df["temperature"] == T]["dH_abs"]
            ),
        }

    dict = dict_kim_dH_abs

    idx_start = 0
    w_MEA = .3
    for T, subdict in dict.items():

        T += 273.15

        subdict["range"] = range(idx_start, idx_start + len(subdict["CO2_loading"]))
        # Start out with a completely unloaded mixture
        x_CO2, x_MEA, x_H2O, n_tot = get_x(0.01, w_MEA)
        blk = params[idx_start]
        blk.flow_mol.fix(n_tot)
        blk.mole_frac_comp["H2O"].fix(x_H2O)
        blk.mole_frac_comp["MEA"].fix(x_MEA)
        blk.mole_frac_comp["CO2"].fix(x_CO2)
        # Not a lot of information about pressure
        blk.pressure.fix(101325)
        blk.temperature.fix(T)
        excess_enth_list = []
        enth_abs_list = []
        enth_abs_list_2 = []
        enth_component_dict = {
            "phys": [],
            "excess": [],
            "MEA_bicarbonate_formation": [],
            "MEA_carbamate_formation_combo": [],
        }

        for k, loading in enumerate(subdict["CO2_loading"]):
            blk_old = blk
            blk = params[idx_start + k + 1]
            x_CO2, x_MEA, x_H2O, n_tot = get_x(loading, w_MEA)
            blk.flow_mol.fix(n_tot)
            blk.mole_frac_comp["H2O"].fix(x_H2O)
            blk.mole_frac_comp["MEA"].fix(x_MEA)
            blk.mole_frac_comp["CO2"].fix(x_CO2)
            # Not a lot of information about pressure
            blk.pressure.fix(101325)
            blk.temperature.fix(T)
            CO2_obj = m.params.CO2
            Hl_f = blk.energy_internal_mol_phase["Liq"]
            Hl_f_2 = blk.enth_mol_phase["Liq"]
            F_f = blk.flow_mol
            Hl_i = blk_old.energy_internal_mol_phase["Liq"]
            Hl_i_2 = blk_old.enth_mol_phase["Liq"]
            F_i = blk_old.flow_mol
            H_ig = NIST.enth_mol_ig_comp.return_expression(blk, CO2_obj, blk.temperature)
            # print(H_ig)

            Ff_CO2 = blk.flow_mol_comp["CO2"]
            Fi_CO2 = blk_old.flow_mol_comp["CO2"]
            dH_abs_expr = -(Hl_f * F_f - Hl_i * F_i - H_ig * (Ff_CO2 - Fi_CO2)) / (Ff_CO2 - Fi_CO2)
            dH_abs_expr_2 = -(Hl_f_2 * F_f - Hl_i_2 * F_i - H_ig * (Ff_CO2 - Fi_CO2)) / (Ff_CO2 - Fi_CO2)

            excess_enth_list.append(-blk.enth_mol_phase_excess["Liq"])
            Hl_i_2 * F_i
            enth_abs_list.append(dH_abs_expr)
            enth_abs_list_2.append(dH_abs_expr_2)

            enth_component_dict["phys"].append(
                -8.314462618 * pyo.units.joule / pyo.units.mol / pyo.units.degK * (
                        CO2_obj.henry_coeff_1
                        - CO2_obj.henry_coeff_2 * T
                        - CO2_obj.henry_coeff_3 * T ** 2
                )
            )
            enth_component_dict["excess"].append(-(blk.enth_mol_phase_excess["Liq"] *
                                                   sum(blk.flow_mol_phase_comp_true["Liq", j] for j in
                                                       blk.components_in_phase("Liq")) -
                                                   blk_old.enth_mol_phase_excess["Liq"] *
                                                   sum(blk.flow_mol_phase_comp_true["Liq", j] for j in
                                                       blk.components_in_phase("Liq"))) /
                                                 (blk.flow_mol_comp["CO2"] - blk_old.flow_mol_comp["CO2"]))
            for rxn in [
                "MEA_bicarbonate_formation",
                "MEA_carbamate_formation_combo",
            ]:
                enth_component_dict[rxn].append(-(blk.dh_rxn[rxn] *
                                                  blk.apparent_inherent_reaction_extent[rxn] -
                                                  blk_old.dh_rxn[rxn] *
                                                  blk_old.apparent_inherent_reaction_extent[rxn]) /
                                                (blk.flow_mol_comp["CO2"] - blk_old.flow_mol_comp["CO2"]))

            # if fit_dH_abs and (loading <= 0.4) and (subdict["dH_abs"][k] <= 130):  # threshold bc of missing vapor phase enthalpy
            #     residual_scale = 50
            #     residual = (dH_abs_expr*1e-3 - subdict["dH_abs"][k])/residual_scale
            #     obj_expr += loss(residual)

        subdict["excess_enth_list"] = excess_enth_list
        subdict["enth_abs_expr"] = enth_abs_list
        subdict["enth_abs_expr_2"] = enth_abs_list_2
        subdict["enth_component_dict"] = enth_component_dict
        idx_start += len(subdict["CO2_loading"]) + 1
    return obj_expr, dict
