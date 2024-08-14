import idaes.core.util.scaling as iscale
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)

from MEA_eNRTL import get_prop_dict
import pandas as pd


def get_estimated_params(m):
    param_list = []
    for rxn_name in ["MEA_bicarbonate_formation", "MEA_carbamate_formation_combo"]:
        rxn_obj = getattr(m.params, "reaction_" + rxn_name)
        param_list.append(rxn_obj.k_eq_coeff_1)
        param_list.append(rxn_obj.k_eq_coeff_2)
        param_list.append(rxn_obj.k_eq_coeff_3)
        # param_list.append(rxn_obj.k_eq_coeff_4)

    # List of molecules that are
    molecules_list = ["H2O",
                      # "MEA",
                      ]
    cations_list = ["MEAH^+"]
    anions_list = ["MEACOO^-",
                   "HCO3^-",
                   ]

    ca_list = []
    for i in range(len(cations_list)):
        for j in range(len(anions_list)):
            ca_list.append(cations_list[i] + ", " + anions_list[j])

    # This is excluding each permutation m-ca -> ca-m
    # Also excluding the m-m' and m'-m interactions since they can be found from
    # binary interactions fitting
    for mi in molecules_list:
        for ca in ca_list:
            param_list.append(m.params.Liq.tau_A[mi, ca])
            param_list.append(m.params.Liq.tau_B[mi, ca])
            # param_list.append(m.params.Liq.tau_A[ca, mi])
            # param_list.append(m.params.Liq.tau_B[ca, mi])
    return param_list


def create_and_scale_params(m):
    # These reactions should be defined in the configuration dictionary found in the MEA_eNRTL file
    # This reaction combination dictionary is reducing the number of total reactions happening in the system
    # and this is the systemitic approach to it rather than explicitly defining the system with only
    # two reactions
    rxn_combinations = {
        "MEA_bicarbonate_formation": {"bicarbonate_formation": 1, "MEA_protonation": 1},
        "MEA_carbamate_formation_combo": {"MEA_carbamate_formation": 1, "MEA_protonation": 1},
    }

    config = get_prop_dict(["H2O", "MEA", "CO2"],
                           excluded_rxns=["H2O_autoionization", "carbonate_formation"],
                           rxn_combinations=rxn_combinations
                           )

    assert "H3O^+" not in config["components"]
    assert "OH^-" not in config["components"]
    params = m.params = GenericParameterBlock(**config)

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
        "MEA": 2,
        "CO2": 5
    }
    mole_frac_true_scaling_factors = {
        "MEA": 1e1,
        "H2O": 2,
        "HCO3^-": 5e3,
        "MEAH^+": 1e1,
        "MEACOO^-": 5e2,
        "CO2": 1e3

    }
    inherent_rxn_scaling_factors = {
        "MEA_bicarbonate_formation": 5e2,
        "MEA_carbamate_formation_combo": 5e2,
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

def create_and_scale_params_solve(m):
    # These reactions should be defined in the configuration dictionary found in the MEA_eNRTL file
    # This reaction combination dictionary is reducing the number of total reactions happening in the system
    # and this is the systemitic approach to it rather than explicitly defining the system with only
    # two reactions
    rxn_combinations = {
        "MEA_bicarbonate_formation": {"bicarbonate_formation": 1, "MEA_protonation": 1},
        "MEA_carbamate_formation_combo": {"MEA_carbamate_formation": 1, "MEA_protonation": 1},
    }

    config = get_prop_dict(["H2O", "MEA", "CO2"],
                           excluded_rxns=["H2O_autoionization", "carbonate_formation"],
                           rxn_combinations=rxn_combinations
                           )

    assert "H3O^+" not in config["components"]
    assert "OH^-" not in config["components"]
    params = m.params = GenericParameterBlock(**config)

    # Gets each parameter that was fitted and fixes its value for the run
    df = pd.read_csv(r'C:\Users\Tanner\Documents\git\MEA\data\Parameters\ParametersOG.csv')
    for i, row in df.iterrows():
        name = row['Name'][7:]
        name = name.split('.')
        value = row['Value']
        i = 0
        obj = params
        while i < len(name)+1:
            if i == len(name):
                obj.fix(value)
                break
            elif obj.name == 'params.Liq':
                parameter_name = name[i][:5]
                mca = name[i][5:].strip("[]/'/'")
                if name[i][6] == 'H' or name[i][6] == 'M':
                    mi, ca1, ca2 = mca.split(',')
                    ca1 = ca1.strip(" /'")
                    ca2 = ca2.strip(" /'")
                    ca = ca1 + ', ' + ca2
                    obj = getattr(obj, parameter_name)[mi, ca]
                else:
                    ca1, ca2, mi = mca.split(',')
                    ca1 = ca1.strip(" /'")
                    ca2 = ca2.strip(" /'")
                    ca = ca1 + ', ' + ca2
                    obj = getattr(obj, parameter_name)[ca, mi]
            else:
                obj = getattr(obj, name[i])
            i += 1


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
        "MEA": 2,
        "CO2": 5
    }
    mole_frac_true_scaling_factors = {
        "MEA": 1e1,
        "H2O": 2,
        "HCO3^-": 5e3,
        "MEAH^+": 1e1,
        "MEACOO^-": 5e2,
        "CO2": 1e3

    }
    inherent_rxn_scaling_factors = {
        "MEA_bicarbonate_formation": 5e2,
        "MEA_carbamate_formation_combo": 5e2,
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
