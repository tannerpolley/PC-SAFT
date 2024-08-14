import pyomo.environ as pyo

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from CESAR_1_eNRTL import get_prop_dict

if __name__ == "__main__":
    m = pyo.ConcreteModel()
    config = get_prop_dict(["H2O", "CO2"])
    params = m.params = GenericParameterBlock(**config)

    m.state = m.params.build_state_block([1], defined_state=True)
    print(f"DOF: {degrees_of_freedom(m)}")
    m.state[1].flow_mol.fix(100)
    m.state[1].mole_frac_comp["H2O"].fix(0.99)
    m.state[1].mole_frac_comp["CO2"].fix(0.01)
    m.state[1].pressure.fix(2e5)
    m.state[1].temperature.fix(350)
    print(f"DOF: {degrees_of_freedom(m)}")

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
        "H2O": 1,
        "CO2": 1e3,
    }
    mole_frac_true_scaling_factors = {
        "CO2": 1e3,
        "H2O": 1,
        # "HCO3^-": 5e5,
        # "H3O^+": 5e5,
        # "CO3^2-": 1e12,
        # "OH^-": 5e11,
        "HCO3^-": 1e3,
        "H3O^+": 1,
        "CO3^2-": 1e3,
        "OH^-": 1,
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

    # params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1e6, index="H2O_autoionization")
    # params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1e6, index="bicarbonate_formation")
    # params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1e12, index="carbonate_formation")
    params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1, index="H2O_autoionization")
    params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1e3, index="bicarbonate_formation")
    params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*1e3, index="carbonate_formation")


    iscale.calculate_scaling_factors(m)


    m.state.initialize(hold_state=False)
    print(pyo.value(m.state[1].log_fug_phase_comp["Liq","CO2"]))
    solver = get_solver(
        "ipopt",
        options={
            # 'bound_push' : 1e-22,
            'nlp_scaling_method': 'user-scaling',
            'linear_solver': 'ma57',
            'OF_ma57_automatic_scaling': 'yes',
            'max_iter': 300,
            'tol': 1e-8,
            'halt_on_ampl_error': 'no',
            # 'mu_strategy': 'monotone',
        }
    )
    solver.solve(m, tee=True)


    print(pyo.value(m.state[1].fug_phase_comp["Liq","CO2"]))
