# -*- coding: utf-8 -*-
"""
Tests for checking that the PC-SAFT functions are working correctly.

@author: Zach Baird
"""
import math
import numpy as np
import pytest
from pcsaft import pcsaft_den, pcsaft_hres, pcsaft_gres, pcsaft_sres
from pcsaft import flashTQ, flashPQ, pcsaft_Hvap
from pcsaft import dielc_water, pcsaft_osmoticC, pcsaft_fugcoef, pcsaft_miac_m, pcsaft_gsolv, pcsaft_lnfugcoef_terms, pcsaft_dielc_eval
from pcsaft import pcsaft_cp, pcsaft_ares, pcsaft_dadt, pcsaft_p, _pcsaft_autodiff_residual_derivatives
from pcsaft import pcsaft_multiphase_lle

import json
from pathlib import Path

from pcsaft.parameters import _resolve_runtime_options, get_prop_dict
from pcsaft.parameters import DATASET_ROOT

def _runtime_to_elec_model(runtime):
    """Convert resolved runtime options to nested elec_model schema for params."""
    radius_to_d_born = {1: 0, 2: 1, 3: 2, 4: 3, 5: 3}
    born_radius_model = int(runtime.get("born_radius_model", 1))
    born_diff_mode = int(runtime.get("born_diff_mode", 0))
    born_model = int(runtime.get("born_model", 1))
    return {
        "rel_perm": {
            "rule": int(runtime.get("dielc_rule", 1)),
            "differential_mode": int(runtime.get("dielc_diff_mode", 0)),
        },
        "hc_model": {
            "dadx_differential_mode": int(runtime.get("hc_dadx_diff_mode", 0)),
        },
        "disp_model": {
            "dadx_differential_mode": int(runtime.get("disp_dadx_diff_mode", 0)),
        },
        "assoc_model": {
            "dadx_differential_mode": int(runtime.get("assoc_dadx_diff_mode", 0)),
        },
        "polar_model": {
            "dadx_differential_mode": int(runtime.get("polar_dadx_diff_mode", 0)),
        },
        "DH_model": {
            "d_ion_mode": int(runtime.get("d_ion_mode", 1)),
            "bjeruum_treatment": bool(runtime.get("bjeruum_treatment", False)),
            "mu_DH_model": {
                "differential_mode": int(runtime.get("mu_DH_diff_mode", 0)),
                "comp_dep_rel_perm": bool(runtime.get("mu_DH_comp_dep_rel_perm", True)),
                "include_sum_term": bool(runtime.get("mu_DH_include_sum_term", True)),
            },
        },
        "include_born_model": bool(runtime.get("include_born_model", born_model != 0)),
        "born_model": {
            "d_Born_mode": int(runtime.get("d_born_mode", radius_to_d_born.get(born_radius_model, 0))),
            "solvation_shell_model": bool(runtime.get("born_solvation_shell_model", born_model == 2)),
            "dielectric_saturation": bool(runtime.get("born_dielectric_saturation", born_model == 2)),
            "bulk_mode": int(runtime.get("born_bulk_mode", runtime.get("born_eps_mode", 0))),
            "mu_born_model": {
                "differential_mode": int(runtime.get("mu_born_diff_mode", 1 if born_diff_mode == 1 else 0)),
                "comp_dep_rel_perm": bool(runtime.get("mu_born_comp_dep_rel_perm", born_diff_mode != 3)),
                "include_sum_term": bool(runtime.get("mu_born_include_sum_term", born_diff_mode != 2)),
                "comp_dep_delta_d": bool(runtime.get("mu_born_comp_dep_delta_d", False)),
            },
        },
    }


def _dataset_file(*parts: str) -> Path:
    return DATASET_ROOT.joinpath(*parts)


def _central_dadt(t, rho, x, params, h=1.0):
    return (pcsaft_ares(t + h, rho, x, params) - pcsaft_ares(t - h, rho, x, params)) / (2.0 * h)


def _neutral_binary_params(dadt_mode='autodiff'):
    return {
        'm': np.asarray([2.4653, 2.8149]),
        's': np.asarray([3.6478, 3.7169]),
        'e': np.asarray([287.35, 285.69]),
        'dadt_differential_mode': dadt_mode,
    }


def test_ares(print_result=False):
    """Test ares with methane/ethane/propane mixture."""
    t = 233.15  # K
    rho = 14330.417110
    x = np.array([0.1, 0.3, 0.6])

    m = np.asarray([1.0000, 1.6069, 2.0020])
    s = np.asarray([3.7039, 3.5206, 3.6184])
    e = np.asarray([150.03, 191.42, 208.11])
    k_ij = np.asarray([
        [0.0, 3.0e-4, 1.15e-2],
        [3.0e-4, 0.0, 5.10e-3],
        [1.15e-2, 5.10e-3, 0.0],
    ])
    params = {"m": m, "s": s, "e": e, "k_ij": k_ij}

    calc = pcsaft_ares(t, rho, x, params)
    ref = -3.54988543593195
    if print_result:
        print('----- ares at 233.15 K -----')
        print(calc)
    assert abs((calc - ref) / ref * 100) < 1e-4

def test_multiphase_lle():
    t = 298.15
    p = 1.0e5
    species = ["H2O-2B-Li", "Na+", "Cl-"]

    canonical = json.loads(
        _dataset_file("2020_Bulow", "user_options.json").read_text(encoding="utf-8")
    )
    runtime = _resolve_runtime_options(canonical)["runtime"]
    runtime["dielc_rule"] = 1
    runtime["dielc_diff_mode"] = 0

    s_water = 2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)
    params = {
        "MW": np.asarray([18.01528e-3, 22.98e-3, 35.45e-3]),
        "m": np.asarray([1.2047, 1.0, 1.0]),
        "s": np.asarray([s_water, 2.8232, 2.7560]),
        "e": np.asarray([353.95, 230.0, 170.0]),
        "e_assoc": np.asarray([2425.7, 0.0, 0.0]),
        "vol_a": np.asarray([0.04509, 0.0, 0.0]),
        "assoc_scheme": ["2B", None, None],
        "dipm": np.asarray([0.0, 0.0, 0.0]),
        "dip_num": np.asarray([1.0, 1.0, 1.0]),
        "z": np.asarray([0.0, 1.0, -1.0]),
        "dielc": np.asarray([78.09, 8.0, 8.0]),
        "d_born": np.asarray([0.0, 3.445, 4.1]),
        "f_solv": np.asarray([1.5, 1.0, 1.0]),
        "k_ij": np.asarray([
            [0.0, 0.0045, -0.25],
            [0.0045, 0.0, 0.317],
            [-0.25, 0.317, 0.0],
        ]),
        "l_ij": np.zeros((3, 3)),
        "k_hb": np.zeros((3, 3)),
        "elec_model": _runtime_to_elec_model(runtime),
        "debug": bool(runtime["debug"]),
    }

    n = np.asarray([1.0 / 0.01801528, 1e-4, 1e-4])
    z_feed = n / np.sum(n)

    out = pcsaft_multiphase_lle(
        t,
        p,
        z_feed,
        params,
        species,
        options={"tpdf_global_trials": 300, "tpdf_local_trials": 120, "tpdf_tol": -1e-6},
    )
    assert "n_phases" in out
    assert "phases" in out
    assert "e_matrix" in out


def test_dielc_rule7_matches_rule1_for_single_solvent_same_ion_dielc():
    species = ['Li+', 'Br-', 'Ethanol']
    x = np.asarray([0.04, 0.04, 0.92], dtype=float)
    params_rule1 = get_prop_dict('2020_Bulow', species, x, 298.15, user_options={'elec_model': {'rel_perm': {'rule': 1}}})
    params_rule7 = get_prop_dict('2020_Bulow', species, x, 298.15, user_options={'elec_model': {'rel_perm': {'rule': 7}}})

    eps1, deps1 = pcsaft_dielc_eval(x, params_rule1)
    eps7, deps7 = pcsaft_dielc_eval(x, params_rule7)

    assert eps1 == pytest.approx(eps7, rel=0.0, abs=1.0e-12)
    assert np.allclose(np.asarray(deps1, dtype=float), np.asarray(deps7, dtype=float), atol=1.0e-12, rtol=0.0)
def test_hres(print_result=False):
    """Test the residual enthalpy function to see if it is working correctly."""
    if print_result:
        print('------ 325 K ------')
        print('\t\t\t PC-SAFT\t Reference \tRelative error')
    t = 325  # K
    p = 101325  # Pa
    # all reference values are from PC-SAFT implemented in Aspen Plus

    # Toluene ----------
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    params = {'m': m, 's': s, 'e': e}

    den = pcsaft_den(t, p, x, params, phase='liq')
    ref = -36809.39  # J mol^-1
    calc = pcsaft_hres(t, den, x, params)
    if print_result:
        print('Toluene, liquid:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol')
    assert abs((calc - ref) / ref * 100) < 1e-2
    den = pcsaft_den(t, p, x, params, phase='vap')
    ref = -362.6777  # J mol^-1
    calc = pcsaft_hres(t, den, x, params)
    if print_result:
        print('Toluene, vapor:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol')
    assert abs((calc - ref) / ref * 100) < 1e-2

    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}

    den = pcsaft_den(t, p, x, params, phase='liq')
    ref = -38924.64  # J mol^-1
    calc = pcsaft_hres(t, den, x, params)
    if print_result:
        print('Acetic acid, liquid:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol')
    assert abs((calc - ref) / ref * 100) < 1e-2
    den = pcsaft_den(t, p, x, params, phase='vap')
    ref = -15393.63  # J mol^-1
    calc = pcsaft_hres(t, den, x, params)
    if print_result:
        print('Acetic acid, vapor:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol')
    assert abs((calc - ref) / ref * 100) < 1e-2

def test_sres(print_result=False):
    """Test the residual entropy function to see if it is working correctly."""
    if print_result:
        print('------ 325 K ------')
        print('\t\t\t PC-SAFT\t Reference \tRelative error')
    t = 325  # K
    p = 101325  # Pa
    # all reference values are from PC-SAFT implemented in Aspen Plus

    # Toluene ----------
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    params = {'m': m, 's': s, 'e': e}

    den = pcsaft_den(t, p, x, params, phase='liq')
    ref = -96.3692  # J mol^-1 K^-1
    calc = pcsaft_sres(t, den, x, params)
    if print_result:
        print('Toluene, liquid:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol/K')
    assert abs((calc - ref) / ref * 100) < 1e-2
    den = pcsaft_den(t, p, x, params, phase='vap')
    ref = -0.71398  # J mol^-1 K^-1
    calc = pcsaft_sres(t, den, x, params)
    if print_result:
        print('Toluene, vapor:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol/K')
    assert abs((calc - ref) / ref * 100) < 1e-2

    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}

    den = pcsaft_den(t, p, x, params, phase='liq')
    ref = -98.1127  # J mol^-1 K^-1
    calc = pcsaft_sres(t, den, x, params)
    if print_result:
        print('Acetic acid, liquid:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol/K')
    assert abs((calc - ref) / ref * 100) < 1e-2
    den = pcsaft_den(t, p, x, params, phase='vap')
    ref = -40.8743  # J mol^-1 K^-1
    calc = pcsaft_sres(t, den, x, params)
    if print_result:
        print('Acetic acid, vapor:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol/K')
    assert abs((calc - ref) / ref * 100) < 1e-2

def test_gres(print_result=False):
    """Test the residual Gibbs energy function to see if it is working correctly."""
    if print_result:
        print('------ 325 K ------')
        print('\t\t\t PC-SAFT\t Reference \tRelative error')
    t = 325  # K
    p = 101325  # Pa
    # all reference values are from PC-SAFT implemented in Aspen Plus

    # Toluene ----------
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    params = {'m': m, 's': s, 'e': e}

    den = pcsaft_den(t, p, x, params, phase='liq')
    ref = -5489.384  # J mol^-1
    calc = pcsaft_gres(t, den, x, params)
    if print_result:
        print('Toluene, liquid:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol')
    assert abs((calc - ref) / ref * 100) < 1e-2
    den = pcsaft_den(t, p, x, params, phase='vap')
    ref = -130.6339  # J mol^-1
    calc = pcsaft_gres(t, den, x, params)
    if print_result:
        print('Toluene, vapor:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol')
    assert abs((calc - ref) / ref * 100) < 1e-2

    # Acetic acid ---------
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}

    den = pcsaft_den(t, p, x, params, phase='liq')
    ref = -7038.004  # J mol^-1
    calc = pcsaft_gres(t, den, x, params)
    if print_result:
        print('Acetic acid, liquid:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol')
    assert abs((calc - ref) / ref * 100) < 1e-2
    den = pcsaft_den(t, p, x, params, phase='vap')
    ref = -2109.459  # J mol^-1
    calc = pcsaft_gres(t, den, x, params)
    if print_result:
        print('Acetic acid, vapor:\t\t', calc, ref, (calc - ref) / ref * 100, 'J/mol')
    assert abs((calc - ref) / ref * 100) < 1e-2

def test_density(print_result=False):
    """Test the density function to see if it is working correctly."""
    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    params = {'m': m, 's': s, 'e': e}

    ref = 9135.590853014008  # source: reference EOS in CoolProp
    calc = pcsaft_den(320, 101325, x, params, phase='liq')
    if print_result:
        print('##########  Test with toluene  ##########')
        print('----- Density at 320 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2

    # Water
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    ref = 55502.5970532902  # source: IAWPS95 EOS
    t = 274
    s = np.asarray([2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}
    calc = pcsaft_den(t, 101325, x, params, phase='liq')
    if print_result:
        print('\n##########  Test with water  ##########')
        print('----- Density at 274 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2

    # Acetic acid
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}

    ref = 17240.  # source: DIPPR correlation
    calc = pcsaft_den(305, 101325, x, params, phase='liq')
    if print_result:
        print('\n##########  Test with acetic acid  ##########')
        print('----- Density at 305 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2

    # Dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    params = {'m': m, 's': s, 'e': e, 'dipm': dpm, 'dip_num': dip_num}

    ref = 16110.  # source: DIPPR correlation
    calc = pcsaft_den(240, 101325, x, params, phase='liq')
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('----- Density at 240 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2

    # Propane
    x = np.asarray([1.])
    m = np.asarray([2.0020])
    s = np.asarray([3.6184])
    e = np.asarray([208.11])
    params = {'m': m, 's': s, 'e': e}

    t = 369.82  # K
    p = 42.473 * 1e5  # Pa
    ref = 5140.3  # source: HEOS equation of state
    calc = pcsaft_den(t, p, x, params, phase='liq')
    if print_result:
        print('\n##########  Test with propane  ##########')
        print('----- Liquid density at {} K and {} Pa -----'.format(t, p))
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 40  # near the critical point, the accuracy is lower

    ref = 4857.2  # source: HEOS equation of state
    calc = pcsaft_den(t, p, x, params, phase='vap')
    if print_result:
        print('----- Vapor density at {} K and {} Pa -----'.format(t, p))
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 40

    # Binary mixture: methanol-cyclohexane
    #0 = methanol, 1 = cyclohexane
    x = np.asarray([0.0550, 0.945])
    m = np.asarray([1.5255, 2.5303])
    s = np.asarray([3.2300, 3.8499])
    e = np.asarray([188.90, 278.11])
    volAB = np.asarray([0.035176, 0.])
    eAB = np.asarray([2899.5, 0.])
    k_ij = np.asarray([[0, 0.051],
                       [0.051, 0]])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij}

    ref = 9506.1  # source: J. Canosa, A. Rodríguez, and J. Tojo, “Liquid−Liquid Equilibrium and Physical Properties of the Ternary Mixture (Dimethyl Carbonate + Methanol + Cyclohexane) at 298.15 K,” J. Chem. Eng. Data, vol. 46, no. 4, pp. 846–850, Jul. 2001.
    calc = pcsaft_den(298.15, 101325, x, params, phase='liq')
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Density at 298.15 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2

    # NaCl in water
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.010579869455908, 0.010579869455908, 0.978840261088184])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                       [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    ref = 55507.23  # source: Rodriguez H.; Soto A.; Arce A.; Khoshkbarchi M.K.: Apparent Molar Volume, Isentropic Compressibility, Refractive Index, and Viscosity of DL-Alanine in Aqueous NaCl Solutions. J.Solution Chem. 32 (2003) 53-63
    t = 298.15  # K
    s[2] = 2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)  # temperature dependent segment diameter for water
    k_ij[0, 2] = -0.007981 * t + 2.37999
    k_ij[2, 0] = -0.007981 * t + 2.37999
    dielc = dielc_water(t)

    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij, 'z': z, 'dielc': np.full(len(m), dielc)}

    calc = pcsaft_den(t, 101325, x, params, phase='liq')
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('----- Density at 298.15 K and 101325 Pa -----')
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2

    # Propane
    x = np.asarray([1.])
    m = np.asarray([2.0020])
    s = np.asarray([3.6184])
    e = np.asarray([208.11])
    params = {'m': m, 's': s, 'e': e}

    t = 85.525  # K
    p = 1.7551e-4  # Pa
    ref = 16621.0  # source: reference EOS in CoolProp
    calc = pcsaft_den(t, p, x, params, phase='liq')
    if print_result:
        print('##########  Test with propane  ##########')
        print('----- Density at {} K and {} Pa -----'.format(t, p))
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2

    t = 85.525  # K
    p = 1.39e-4  # Pa
    ref = 1.9547e-7  # source: reference EOS in CoolProp
    calc = pcsaft_den(t, p, x, params, phase='vap')
    if print_result:
        print('----- Density at {} K and {} Pa -----'.format(t, p))
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2

    t = 293  # K
    p = 833240  # Pa
    ref = 11346.0  # source: reference EOS in CoolProp
    calc = pcsaft_den(t, p, x, params, phase='liq')
    if print_result:
        print('----- Density at {} K and {} Pa -----'.format(t, p))
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2

    t = 430  # K
    p = 2000000  # Pa
    ref = 623.59  # source: reference EOS in CoolProp
    calc = pcsaft_den(t, p, x, params, phase='liq')
    if print_result:
        print('----- Density at {} K and {} Pa -----'.format(t, p))
        print('    Reference:', ref, 'mol m^-3')
        print('    PC-SAFT:', calc, 'mol m^-3')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2

def test_indexes(print_result=False):
    '''
    Check that the properties of a pure compound are the same regardless of
    whether parameters for additional compounds are included.
    '''
    # Binary mixture: water-acetic acid
    # only parameters for acetic acid
    x = np.asarray([1.])
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}

    ref = 193261.515187248  # source: DIPPR correlation
    t = 413.5385
    rho = 15107.481234283325
    fugcoef1 = pcsaft_fugcoef(t, rho, x, params)

    # same composition, but with mixture parameters
    #0 = water, 1 = acetic acid
    m = np.asarray([1.2047, 1.3403])
    s = np.asarray([0, 3.8582])
    e = np.asarray([353.95, 211.59])
    volAB = np.asarray([0.0451, 0.075550])
    eAB = np.asarray([2425.67, 3044.4])
    k_ij = np.asarray([[0, -0.127],
                       [-0.127, 0]])

    x = np.asarray([0, 1])
    s[0] = 3.8395 + 1.2828 * np.exp(-0.0074944 * t) - 1.3939 * np.exp(-0.00056029 * t)
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij}
    fugcoef_mix = pcsaft_fugcoef(t, rho, x, params)
    if print_result:
        print('\n##########  Test with acetic acid  ##########')
        print('pure fugcoef:', fugcoef1[0])
        print('mix fugcoef:', fugcoef_mix[1])
        print('deviation', (fugcoef_mix[1] - fugcoef1[0]) / fugcoef1[0] * 100, '%')
    assert abs((fugcoef_mix[1] - fugcoef1) / fugcoef1 * 100) < 1e-1

    # Binary mixture: water-furfural
    # only parameters for furfural
    x = np.asarray([1.])
    m = np.asarray([3.9731])
    s = np.asarray([3.0551])
    e = np.asarray([259.15])
    dipm = np.asarray([3.6])
    dip_num = np.asarray([1])
    params = {'m': m, 's': s, 'e': e, 'dipm': dipm, 'dip_num': dip_num}

    t = 400  # K
    p = 34914.37778265716  # Pa
    rho = 10899.584105341197
    fugcoef1 = pcsaft_fugcoef(t, rho, x, params)

    # same composition, but with mixture parameters
    #0 = water, 1 = furfural
    m = np.asarray([1.2047, 3.9731])
    s = np.asarray([0, 3.0551])
    e = np.asarray([353.95, 259.15])
    volAB = np.asarray([0.0451, 0.0451])
    eAB = np.asarray([2425.67, 0])
    dipm = np.asarray([0, 3.6])
    dip_num = np.asarray([0, 1])
    k_ij = np.asarray([[0, -0.027],
                       [-0.027, 0]])

    x = np.asarray([0, 1])
    s[0] = 3.8395 + 1.2828 * np.exp(-0.0074944 * t) - 1.3939 * np.exp(-0.00056029 * t)
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'dipm': dipm, 'dip_num': dip_num, 'k_ij': k_ij}
    fugcoef_mix = pcsaft_fugcoef(t, rho, x, params)
    if print_result:
        print('\n##########  Test with furfural  ##########')
        print('pure fugcoef:', fugcoef1[0])
        print('mix fugcoef:', fugcoef_mix[1])
        print('deviation', (fugcoef_mix[1] - fugcoef1[0]) / fugcoef1[0] * 100, '%')
    assert abs((fugcoef_mix[1] - fugcoef1) / fugcoef1 * 100) < 1e-6

    # Mixture: NaCl in water with random 4th component
    # only parameters for water
    x = np.asarray([1.])
    m = np.asarray([1.2047])
    s = np.asarray([0.])
    e = np.asarray([353.9449])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    t = 298.15  # K
    s[0] = 3.8395 + 1.2828 * np.exp(-0.0074944 * t) - 1.3939 * np.exp(-0.00056029 * t)
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}

    p = 3153.417688548272  # Pa
    rho = 55320.89616248148
    fugcoef1 = pcsaft_fugcoef(t, rho, x, params)

    # same composition, but with mixture parameters
    # Mixture: NaCl in water with random 4th component
    # 0 = Na+, 1 = Cl-, 2 = H2O, 3 = DIMETHOXYMETHANE
    x = np.asarray([0, 0, 1, 0])
    m = np.asarray([1, 1, 1.2047, 2.8454])
    s = np.asarray([2.8232, 2.7599589, 0., 3.4017])
    e = np.asarray([230.00, 170.00, 353.9449, 234.02])
    volAB = np.asarray([0, 0, 0.0451, 0.0451])
    eAB = np.asarray([0, 0, 2425.67, 0])
    dipm = np.asarray([0, 0, 0, 1.2])
    dip_num = np.asarray([0, 0, 0, 1])
    k_ij = np.asarray([[0, 0.317, 0, 0],
                       [0.317, 0, -0.25, 0],
                       [0, -0.25, 0, 0],
                       [0, 0, 0, 0]])
    k_ij[0, 2] = -0.007981 * t + 2.37999
    k_ij[2, 0] = -0.007981 * t + 2.37999
    z = np.asarray([1., -1., 0., 0])
    dielc = dielc_water(t)

    s[2] = 3.8395 + 1.2828 * np.exp(-0.0074944 * t) - 1.3939 * np.exp(-0.00056029 * t)
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'dipm': dipm, 'dip_num': dip_num, 'k_ij': k_ij, 'z': z,
              'dielc': np.full(len(m), dielc)}
    fugcoef_mix = pcsaft_fugcoef(t, rho, x, params)
    if print_result:
        print('\n##########  Test with water  ##########')
        print('pure fugcoef:', fugcoef1[0])
        print('mix fugcoef:', fugcoef_mix[2])
        print('deviation', (fugcoef_mix[2] - fugcoef1[0]) / fugcoef1[0] * 100, '%')
    assert abs((fugcoef_mix[2] - fugcoef1) / fugcoef1 * 100) < 1e-1

def test_flashTQ(print_result=False):
    """Test the flashTQ function to see if it is working correctly."""
    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    params = {'m': m, 's': s, 'e': e}

    ref = 3255792.76201971  # source: reference EOS in CoolProp
    t = 572.6667
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('##########  Test with toluene  ##########')
        print('----- Vapor pressure at 572.7 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    # Water
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    ref = 67171.754576141  # source: IAWPS95 EOS
    t = 362
    s = np.asarray([2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('\n##########  Test with water  ##########')
        print('----- Vapor pressure at 362 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    # Acetic acid
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}

    ref = 193261.515187248  # source: DIPPR correlation
    t = 413.5385
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('\n##########  Test with acetic acid  ##########')
        print('----- Vapor pressure at 413.5 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    # Dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    params = {'m': m, 's': s, 'e': e, 'dipm': dpm, 'dip_num': dip_num}

    ref = 625100.  # source: DIPPR correlation
    t = 300
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('----- Vapor pressure at 300 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    # Propane
    x = np.asarray([1.])
    m = np.asarray([2.0020])
    s = np.asarray([3.6184])
    e = np.asarray([208.11])
    params = {'m': m, 's': s, 'e': e}

    ref = 1.7551e-4  # source: reference EOS in CoolProp
    t = 85.525
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('##########  Test with propane  ##########')
        print('----- Vapor pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 10

    ref = 8.3324e5  # source: reference EOS in CoolProp
    t = 293
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('----- Vapor pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    ref = 42.477e5  # source: reference EOS in CoolProp
    t = 369.82
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('----- Vapor pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    # Binary mixture: methane-benzene
    #0 = methane, 1 = benzene
    m = np.asarray([1.0000, 2.4653])
    s = np.asarray([3.7039, 3.6478])
    e = np.asarray([150.03, 287.35])
    k_ij = np.asarray([[0, 0.037],
                       [0.037, 0]])
    params = {'m': m, 's': s, 'e': e, 'k_ij': k_ij}

    x = np.asarray([0.0252, 0.9748])
    t = 421.05
    ref = 1986983.25  # source: H.-M. Lin, H. M. Sebastian, J. J. Simnick, and K.-C. Chao, “Gas-liquid equilibrium in binary mixtures of methane with N-decane, benzene, and toluene,” J. Chem. Eng. Data, vol. 24, no. 2, pp. 146–149, Apr. 1979.
    xv_ref = np.asarray([0.6516, 0.3484])
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('\n##########  Test with methane-benzene mixture  ##########')
        print('----- Bubble point pressure at %s K -----' % t)
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv - xv_ref) / xv_ref * 100)
    assert abs((calc - ref) / ref * 100) < 10
    assert np.all(abs((xv - xv_ref) / xv_ref * 100) < 10)

    # This doesn't pass yet because the code doesn't yet have a way to get a good enough initial guess of the equilibrium ratio between vapor and liquid composition (k factors)
    # x = np.asarray([0.119,0.881])
    # t = 348.15
    # ref = 6691000 # source: Hughes TJ, Kandil ME, Graham BF, Marsh KN, Huang SH, May EF. Phase equilibrium measurements of (methane+ benzene) and (methane+ methylbenzene) at temperatures from (188 to 348) K and pressures to 13 MPa. The Journal of Chemical Thermodynamics. 2015 Jun 1;85:141-7.
    # xv_ref = np.asarray([0.9675,0.0325])
    # calc, xl, xv = flashTQ(t, 0, x, params)
    # if print_result:
    #     print('\n##########  Test with methane-benzene mixture  ##########')
    #     print('----- Bubble point pressure at %s K -----' % t)
    #     print('    Reference:', ref, 'Pa')
    #     print('    PC-SAFT:', calc, 'Pa')
    #     print('    Relative deviation:', (calc-ref)/ref*100, '%')
    #     print('    Vapor composition (reference):', xv_ref)
    #     print('    Vapor composition (PC-SAFT):', xv)
    #     print('    Vapor composition relative deviation:', (xv-xv_ref)/xv_ref*100)
    # assert abs((calc-ref)/ref*100) < 10
    # assert np.all(abs((xv-xv_ref)/xv_ref*100) < 10)

    # # Binary mixture: hydrogen-toluene
    # # This does not pass yet. Although our values for this system match those given by Joachim Gross's code, neither
    # # code is able to accurately match the literature data.
    # #0 = hydrogen, 1 = toluene
    # x0 = 0.037
    # x = np.asarray([x0, 1-x0])
    # m = np.asarray([1.0000, 2.8149])
    # s = np.asarray([2.9860, 3.7169])
    # e = np.asarray([19.2775, 285.69])
    # k_ij = np.asarray([[0, 0],
    #                    [0, 0]])
    # params = {'m':m, 's':s, 'e':e, 'k_ij':k_ij}
    #
    # t = 501.6 # K
    # ref = 50.0 * 1e5 # Pa, source: Lin, H.-M.; Sebastian,H.M.; Chao,K.-C.; J. Chem. Engng. Data. 1980, 25, 252-257.
    # xv_ref = np.asarray([0.9648, 0.0352])
    # calc, xl, xv = flashTQ(t, 0, x, params, p_guess=ref)
    # if print_result:
    #     print('\n##########  Test with hydrogen-toluene mixture  ##########')
    #     print('----- Bubble point pressure at %s K -----' % t)
    #     print('    Reference:', ref, 'Pa')
    #     print('    PC-SAFT:', calc, 'Pa')
    #     print('    Relative deviation:', (calc-ref)/ref*100, '%')
    #     print('    Vapor composition (reference):', xv_ref)
    #     print('    Vapor composition (PC-SAFT):', xv)
    #     print('    Vapor composition relative deviation:', (xv-xv_ref)/xv_ref*100)
    # assert abs((calc-ref)/ref*100) < 10
    # assert np.all(abs((xv-xv_ref)/xv_ref*100) < 10)

    # # Binary mixture: hydrogen-hexadecane
    # # This does not pass yet. Although our values for this system match those given by Joachim Gross's code, neither
    # # code is able to accurately match the literature data.
    # #0 = hydrogen, 1 = hexadecane
    # x = np.asarray([0.0407, 0.9593])
    # m = np.asarray([1.0000, 6.6485])
    # s = np.asarray([2.9860, 3.9552])
    # e = np.asarray([19.2775, 254.70])
    # k_ij = np.asarray([[0, -0.1],
    #                    [-0.1, 0]])
    # params = {'m':m, 's':s, 'e':e, 'k_ij':k_ij}
    #
    # t = 542.25 # K
    # ref = 2.009 * 1000000 # Pa, source: Lin, H.-M.; Sebastian,H.M.; Chao,K.-C.; J. Chem. Engng. Data. 1980, 25, 252-257.
    # xv_ref = np.asarray([0.9648, 0.0352])
    # calc, xl, xv = flashTQ(t, 0, x, params, p_guess=0.237*ref)
    # if print_result:
    #     print('\n##########  Test with hydrogen-hexadecane mixture  ##########')
    #     print('----- Bubble point pressure at %s K -----' % t)
    #     print('    Reference:', ref, 'Pa')
    #     print('    PC-SAFT:', calc, 'Pa')
    #     print('    Relative deviation:', (calc-ref)/ref*100, '%')
    #     print('    Vapor composition (reference):', xv_ref)
    #     print('    Vapor composition (PC-SAFT):', xv)
    #     print('    Vapor composition relative deviation:', (xv-xv_ref)/xv_ref*100)
    # assert abs((calc-ref)/ref*100) < 10
    # assert np.all(abs((xv-xv_ref)/xv_ref*100) < 10)

    # Binary mixture: methanol-cyclohexane
    #0 = methanol, 1 = cyclohexane
    x = np.asarray([0.3, 0.7])
    m = np.asarray([1.5255, 2.5303])
    s = np.asarray([3.2300, 3.8499])
    e = np.asarray([188.90, 278.11])
    volAB = np.asarray([0.035176, 0.])
    eAB = np.asarray([2899.5, 0.])
    k_ij = np.asarray([[0, 0.051],
                       [0.051, 0]])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij}

    # bubble point
    ref = 101330  # source: Marinichev A. N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    t = 327.48
    xv_ref = np.asarray([0.59400, 0.40600])
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Bubble point pressure at 327.48 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv - xv_ref) / xv_ref * 100)
    assert abs((calc - ref) / ref * 100) < 10
    assert np.all(abs((xv - xv_ref) / xv_ref * 100) < 10)

    # dew point
    x = np.asarray([0.59400, 0.40600])
    ref = 101330  # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    t = 327.48
    xl_ref = np.asarray([0.3, 0.7])
    calc, xl, xv = flashTQ(t, 1, x, params)
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Dew point pressure at 327.48 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
        print('    Liquid composition (reference):', xl_ref)
        print('    Liquid composition (PC-SAFT):', xl)
        print('    Liquid composition relative deviation:', (xl - xl_ref) / xl_ref * 100)
    assert abs((calc - ref) / ref * 100) < 10
    assert np.all(abs((xl - xl_ref) / xl_ref * 100) < 15)

    # Binary mixture: chloroform-ethanol
    #0 = chloroform, 1 = ethanol
    x = np.asarray([0.3607, 0.6393])
    m = np.asarray([2.5313, 2.3827])
    s = np.asarray([3.4608, 3.1771])
    e = np.asarray([269.47, 198.24])
    volAB = np.asarray([0.032384, 0.032384])
    eAB = np.asarray([0, 2653.4])
    dipm = np.asarray([1.04, 0.])
    dipnum = np.asarray([1, 0.])
    k_ij = np.asarray([[0, 0],
                       [0, 0]])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'dipm': dipm, 'dip_num': dipnum, 'k_ij': k_ij}

    ref = 101330  # source: Chen GH, Wang Q, Ma ZM, Yan XH, Han SJ. Phase equilibria at superatmospheric pressures for systems containing halohydrocarbon, aromatic hydrocarbon, and alcohol. Journal of Chemical and Engineering Data. 1995 Mar;40(2):361-6.
    t = 337.03
    xv_ref = np.asarray([0.6127, 0.3873])
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('\n##########  Test with chloroform-ethanol mixture  ##########')
        print('----- Bubble point pressure at 327.48 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv - xv_ref) / xv_ref * 100)
    assert abs((calc - ref) / ref * 100) < 10
    assert np.all(abs((xv - xv_ref) / xv_ref * 100) < 10)

    # Binary mixture: water-acetic acid
    #0 = water, 1 = acetic acid
    m = np.asarray([1.2047, 1.3403])
    s = np.asarray([0, 3.8582])
    e = np.asarray([353.95, 211.59])
    volAB = np.asarray([0.0451, 0.075550])
    eAB = np.asarray([2425.67, 3044.4])
    k_ij = np.asarray([[0, -0.127],
                       [-0.127, 0]])

    xl_ref = np.asarray([0.9898662364, 0.0101337636])
    t = 403.574
    s[0] = 3.8395 + 1.2828 * np.exp(-0.0074944 * t) - 1.3939 * np.exp(-0.00056029 * t)
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij}
    ref = 273722.  # source: Othmer, D. F.; Silvis, S. J.; Spiel, A. Ind. Eng. Chem., 1952, 44, 1864-72 Composition of vapors from boiling binary solutions pressure equilibrium still for studying water - acetic acid system
    xv_ref = np.asarray([0.9923666645, 0.0076333355])
    calc, xl, xv = flashTQ(t, 0, xl_ref, params)
    if print_result:
        print('\n##########  Test with water-acetic acid mixture  ##########')
        print('----- Bubble point pressure at %s K -----' % t)
        print('    Liquid composition:', xl_ref)
        print('    Reference pressure:', ref, 'Pa')
        print('    PC-SAFT pressure:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv - xv_ref) / xv_ref * 100)
    assert abs((calc - ref) / ref * 100) < 10
    assert np.all(abs((xv - xv_ref) / xv_ref * 100) < 15)

    xl_ref = np.asarray([0.2691800943, 0.7308199057])
    t = 372.774
    s[0] = 3.8395 + 1.2828 * np.exp(-0.0074944 * t) - 1.3939 * np.exp(-0.00056029 * t)
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij}
    ref = 74463.  # source: Freeman, J. R.; Wilson, G. M. AIChE Symp. Ser., 1985, 81, 14-25 High temperature vapor-liquid equilibrium measurements on acetic acid/water mixtures
    xv_ref = np.asarray([0.3878269411, 0.6121730589])
    calc, xl, xv = flashTQ(t, 0, xl_ref, params)
    if print_result:
        print('----- Bubble point pressure at %s K -----' % t)
        print('    Liquid composition:', xl_ref)
        print('    Reference pressure:', ref, 'Pa')
        print('    PC-SAFT pressure:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv - xv_ref) / xv_ref * 100)
    assert abs((calc - ref) / ref * 100) < 10
    assert np.all(abs((xv - xv_ref) / xv_ref * 100) < 15)

    # NaCl in water
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.0907304774758426, 0.0907304774758426, 0.818539045048315])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                       [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    ref = 2393.8  # average of repeat data points from source: A. Apelblat and E. Korin, “The vapour pressures of saturated aqueous solutions of sodium chloride, sodium bromide, sodium nitrate, sodium nitrite, potassium iodate, and rubidium chloride at temperatures from 227 K to 323 K,” J. Chem. Thermodyn., vol. 30, no. 1, pp. 59–71, Jan. 1998. (Solubility calculated using equation from Yaws, Carl L.. (2008). Yaws' Handbook of Properties for Environmental and Green Engineering.)
    t = 298.15  # K
    s[2] = 2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)  # temperature dependent segment diameter for water
    k_ij[0, 2] = -0.007981 * t + 2.37999
    k_ij[2, 0] = -0.007981 * t + 2.37999
    dielc = dielc_water(t)

    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij, 'z': z, 'dielc': np.full(len(m), dielc)}

    xv_guess = np.asarray([0., 0., 1.])
    calc, xl, xv = flashTQ(t, 0, x, params)
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('----- Bubble point pressure at 298.15 K -----')
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 25

def test_flashPQ(print_result=False):
    """Test the flashPQ function to see if it is working correctly."""
    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    params = {'m': m, 's': s, 'e': e}

    p = 3255792.76201971  # source: reference EOS in CoolProp
    ref = 572.6667
    calc, xl, xv = flashPQ(p, 0, x, params)
    if print_result:
        print('##########  Test with toluene  ##########')
        print('----- Vapor pressure at {} Pa -----'.format(p))
        print('    Reference:', ref, 'K')
        print('    PC-SAFT:', calc, 'K')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    # Binary mixture: methanol-cyclohexane
    #0 = methanol, 1 = cyclohexane
    x = np.asarray([0.3, 0.7])
    m = np.asarray([1.5255, 2.5303])
    s = np.asarray([3.2300, 3.8499])
    e = np.asarray([188.90, 278.11])
    volAB = np.asarray([0.035176, 0.])
    eAB = np.asarray([2899.5, 0.])
    k_ij = np.asarray([[0, 0.051],
                       [0.051, 0]])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij}

    # bubble point
    p = 101330
    ref = 327.48  # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    xv_ref = np.asarray([0.59400, 0.40600])
    calc, xl, xv = flashPQ(p, 0, x, params)
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Bubble point temperature at 101330 Pa -----')
        print('    Reference:', ref, 'K')
        print('    PC-SAFT:', calc, 'K')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
        print('    Vapor composition (reference):', xv_ref)
        print('    Vapor composition (PC-SAFT):', xv)
        print('    Vapor composition relative deviation:', (xv - xv_ref) / xv_ref * 100)
    assert abs((calc - ref) / ref * 100) < 1
    assert np.all(abs((xv - xv_ref) / xv_ref * 100) < 10)

    # dew point
    x = np.asarray([0.59400, 0.40600])
    p = 101330
    ref = 327.48  # source: Marinichev A.N.; Susarev M.P.: Investigation of Liquid-Vapor Equilibrium in the System Methanol-Cyclohexane at 35, 45 and 55°C and 760 mm Hg. J.Appl.Chem.USSR 38 (1965) 1582-1584
    xl_ref = np.asarray([0.3, 0.7])
    calc, xl, xv = flashPQ(p, 1, x, params)
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Dew point temperature at 101330 Pa -----')
        print('    Reference:', ref, 'K')
        print('    PC-SAFT:', calc, 'K')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
        print('    Liquid composition (reference):', xl_ref)
        print('    Liquid composition (PC-SAFT):', xl)
        print('    Liquid composition relative deviation:', (xl - xl_ref) / xl_ref * 100)
    assert abs((calc - ref) / ref * 100) < 1
    assert np.all(abs((xl - xl_ref) / xl_ref * 100) < 20)

    # NaCl in water
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.0907304774758426, 0.0907304774758426, 0.818539045048315])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                       [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    p = 2393.8  # Pa
    ref = 298.15  # K, average of repeat data points from source: A. Apelblat and E. Korin, “The vapor pressures of saturated aqueous solutions of sodium chloride, sodium bromide, sodium nitrate, sodium nitrite, potassium iodate, and rubidium chloride at temperatures from 227 K to 323 K,” J. Chem. Thermodyn., vol. 30, no. 1, pp. 59–71, Jan. 1998. (Solubility calculated using equation from Yaws, Carl L.. (2008). Yaws' Handbook of Properties for Environmental and Green Engineering.)
    s[2] = 3.8395 + 1.2828 * np.exp(-0.0074944 * ref) - 1.3939 * np.exp(
        -0.00056029 * ref)  # temperature dependent segment diameter for water
    k_ij[0, 2] = -0.007981 * ref + 2.37999
    k_ij[2, 0] = -0.007981 * ref + 2.37999
    dielc = dielc_water(ref)

    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij, 'z': z, 'dielc': np.full(len(m), dielc)}

    xv_guess = np.asarray([0., 0., 1.])
    calc, xl, xv = flashPQ(p, 0, x, params)
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('----- Bubble point temperature at 2393.8 Pa -----')
        print('    Reference:', ref, 'K')
        print('    PC-SAFT:', calc, 'K')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 1.5

def test_osmoticC(print_result=False):
    """Test the function for calculating osmotic coefficients to see if it is working correctly."""
    x = np.asarray([0.0629838206, 0.0629838206, 0.8740323588])
    t = 293.15  # K

    canonical = json.loads(
        _dataset_file("2014_Held", "user_options.json").read_text(encoding="utf-8")
    )
    runtime = _resolve_runtime_options(canonical)["runtime"]

    s_water = 2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)
    k_na_h2o = -0.007981 * t + 2.37999
    params = {
        "MW": np.asarray([22.98e-3, 35.45e-3, 18.01528e-3]),
        "m": np.asarray([1.0, 1.0, 1.2047]),
        "s": np.asarray([2.8232, 2.7560, s_water]),
        "e": np.asarray([230.0, 170.0, 353.95]),
        "e_assoc": np.asarray([0.0, 0.0, 2425.7]),
        "vol_a": np.asarray([0.0, 0.0, 0.0451]),
        "assoc_scheme": [None, None, "2B"],
        "dipm": np.asarray([0.0, 0.0, 0.0]),
        "dip_num": np.asarray([1.0, 1.0, 1.0]),
        "z": np.asarray([1.0, -1.0, 0.0]),
        "dielc": np.asarray([8.0, 8.0, 78.09]),
        "d_born": np.asarray([3.445, 4.1, 0.0]),
        "f_solv": np.asarray([1.0, 1.0, 1.5]),
        "k_ij": np.asarray([
            [0.0, 0.317, k_na_h2o],
            [0.317, 0.0, -0.25],
            [k_na_h2o, -0.25, 0.0],
        ]),
        "l_ij": np.zeros((3, 3)),
        "k_hb": np.zeros((3, 3)),
        "elec_model": _runtime_to_elec_model(runtime),
        "debug": bool(runtime["debug"]),
    }

    ref = 1.116  # source: R. A. Robinson and R. H. Stokes, Electrolyte Solutions: Second Revised Edition. Dover Publications, 1959.

    rho = pcsaft_den(t, 2339.3, x, params, phase='liq')
    result = pcsaft_osmoticC(t, rho, x, params)
    calc = result[0]
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('----- Osmotic coefficient at 293.15 K -----')
        print('    Reference:', ref)
        print('    PC-SAFT:', calc)
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 2


def test_miac_m(print_result=False):
    """Test molality-scale MIAC (structure mirrors osmotic test)."""
    t = 298.15  # K
    p = 101325
    m_salt = 1.0
    species = ['Na+', 'Br-', 'Methanol']

    canonical = json.loads(
        _dataset_file("2025_Figiel", "user_options.json").read_text(encoding="utf-8")
    )
    runtime = _resolve_runtime_options(canonical)["runtime"]
    runtime["dielc_diff_mode"] = 0
    runtime["debug"] = True

    params = {
        "MW": np.asarray([22.98e-3, 79.90e-3, 32.04e-3]),
        "m": np.asarray([1.0, 1.0, 1.5255]),
        "s": np.asarray([2.8232, 3.0707, 3.2300]),
        "e": np.asarray([230.0, 190.0, 188.90]),
        "e_assoc": np.asarray([0.0, 0.0, 2899.5]),
        "vol_a": np.asarray([0.0, 0.0, 0.03518]),
        "assoc_scheme": [None, None, "2B"],
        "dipm": np.asarray([0.0, 0.0, 0.0]),
        "dip_num": np.asarray([1.0, 1.0, 1.0]),
        "z": np.asarray([1.0, -1.0, 0.0]),
        "dielc": np.asarray([8.0, 8.0, 33.05]),
        "d_born": np.asarray([3.445, 4.48, 0.0]),
        "f_solv": np.asarray([1.0, 1.0, 1.4]),
        "k_ij": np.asarray([
            [0.0, 0.65, -0.25],
            [0.65, 0.0, 0.15],
            [-0.25, 0.15, 0.0],
        ]),
        "l_ij": np.zeros((3, 3)),
        "k_hb": np.zeros((3, 3)),
        "elec_model": _runtime_to_elec_model(runtime),
        "debug": bool(runtime["debug"]),
    }

    n = np.asarray([m_salt, m_salt, 1.0 / 0.03204])
    x = n / np.sum(n)
    rho = pcsaft_den(t, p, x, params, phase='liq')

    ref = 0.38
    calc = pcsaft_miac_m(t, rho, x, params, species=species)['Na+Br-']
    if print_result:
        print('\n##########  Test with NaBr in methanol ##########')
        print('----- MIAC_m at 298.15 K -----')
        print('    Reference:', ref)
        print('    PC-SAFT:', calc)
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert np.all(np.isfinite(calc))


def test_miac_m_mixed_solvent_reference_preserves_solvent_blend():
    """Mixed-solvent MIAC_m must use the salt-free solvent blend at infinite dilution."""
    t = 298.15
    p = 101325.0
    species = ['Na+', 'Br-', 'H2O-2B-NaCl', 'Methanol']
    m_salt = 0.5
    comp = {'water': 0.4, 'methanol': 0.6}

    mw_mix = comp['water'] * 18.01528e-3 + comp['methanol'] * 32.04e-3
    n_solv = 1.0 / mw_mix
    n = np.asarray([m_salt, m_salt, comp['water'] * n_solv, comp['methanol'] * n_solv], dtype=float)
    x = n / np.sum(n)

    params = get_prop_dict('2025_Figiel', species, x, t)
    rho = pcsaft_den(t, p, x, params, phase='liq')
    calc = pcsaft_miac_m(t, rho, x, params, species=species)['Na+Br-']

    fugcoef = np.asarray(pcsaft_fugcoef(t, rho, x, params), dtype=float)
    z = np.asarray(params['z'], dtype=float)
    idx_sol = np.where(np.abs(z) < 1e-12)[0]
    idx_cat = np.where(z > 0)[0]
    idx_an = np.where(z < 0)[0]

    eps = 1e-12
    x_inf = np.full_like(x, eps)
    solvent_ref = np.asarray(x[idx_sol], dtype=float)
    solvent_ref /= np.sum(solvent_ref)
    solvent_budget = max(1.0 - eps * (len(x) - len(idx_sol)), eps * len(idx_sol))
    x_inf[idx_sol] = solvent_ref * solvent_budget
    x_inf /= np.sum(x_inf)

    rho_inf = pcsaft_den(t, p, x_inf, params, phase='liq')
    fugcoef_inf = np.asarray(pcsaft_fugcoef(t, rho_inf, x_inf, params), dtype=float)
    gamma_i = fugcoef / fugcoef_inf

    mw = np.asarray(params['MW'], dtype=float)
    mass_solvent = float(np.sum(x[idx_sol] * mw[idx_sol]))
    mass_neutral = x[idx_sol] * mw[idx_sol]
    w_sf = mass_neutral / mass_neutral.sum()
    m_solvent_mix = 1.0 / np.sum(w_sf / mw[idx_sol])

    ic = idx_cat[0]
    ia = idx_an[0]
    n_salt = 0.5 * (x[ic] + x[ia])
    m_mix_salt = n_salt / mass_solvent
    ln_gamma_pm = 0.5 * (math.log(gamma_i[ic]) + math.log(gamma_i[ia]))
    expected = math.exp(ln_gamma_pm) / (1.0 + m_solvent_mix * m_mix_salt * 2.0)

    assert np.isfinite(calc)
    assert np.isfinite(expected)
    assert abs(calc - expected) < 1e-10


def test_figiel2025_mixed_solvent_any_solvent_sources_are_used():
    t = 298.15
    species = ['Na+', 'Br-', 'H2O-2B-NaCl', 'Methanol']
    x = np.asarray([1e-8, 1e-8, 0.5, 0.5], dtype=float)

    params = get_prop_dict('2025_Figiel', species, x, t)

    assert params['solvated_ion_diameter_mixing_rule'] is False
    assert params['ion_dispersion_mixing_rule'] is True
    assert params.get('mixed_ion_sigma_applied') is None
    assert params.get('mixed_ion_dispersion_applied') is True
    assert params.get('mixed_ion_dispersion_sources') == {'pure/any_solvent.csv': 2.0}

    forced = get_prop_dict(
        '2025_Figiel',
        species,
        x,
        t,
        user_options={'solvated_ion_diameter_mixing_rule': True, 'ion_dispersion_mixing_rule': True},
    )

    assert forced['solvated_ion_diameter_mixing_rule'] is True
    assert forced['ion_dispersion_mixing_rule'] is True
    assert forced.get('mixed_ion_sigma_applied') is True
    assert forced.get('mixed_ion_dispersion_applied') is True
    assert forced.get('mixed_ion_sigma_sources') == {'pure/any_solvent.csv': 2.0}
    assert forced.get('mixed_ion_dispersion_sources') == {'pure/any_solvent.csv': 2.0}


def test_lnfugcoef_terms_structure():
    """Validate structured per-term ln fugacity contributions API."""
    t = 298.15
    p = 101325

    canonical = json.loads(
        _dataset_file("2025_Figiel", "user_options.json").read_text(encoding="utf-8")
    )
    runtime = _resolve_runtime_options(canonical)["runtime"]
    runtime["dielc_diff_mode"] = 0

    params = {
        "MW": np.asarray([22.98e-3, 79.90e-3, 32.04e-3]),
        "m": np.asarray([1.0, 1.0, 1.5255]),
        "s": np.asarray([2.8232, 3.0707, 3.2300]),
        "e": np.asarray([230.0, 190.0, 188.90]),
        "e_assoc": np.asarray([0.0, 0.0, 2899.5]),
        "vol_a": np.asarray([0.0, 0.0, 0.03518]),
        "assoc_scheme": [None, None, "2B"],
        "dipm": np.asarray([0.0, 0.0, 0.0]),
        "dip_num": np.asarray([1.0, 1.0, 1.0]),
        "z": np.asarray([1.0, -1.0, 0.0]),
        "dielc": np.asarray([8.0, 8.0, 33.05]),
        "d_born": np.asarray([3.445, 4.48, 0.0]),
        "f_solv": np.asarray([1.0, 1.0, 1.4]),
        "k_ij": np.asarray([
            [0.0, 0.65, -0.25],
            [0.65, 0.0, 0.15],
            [-0.25, 0.15, 0.0],
        ]),
        "l_ij": np.zeros((3, 3)),
        "k_hb": np.zeros((3, 3)),
        "elec_model": _runtime_to_elec_model(runtime),
        "debug": False,
    }

    n = np.asarray([1.0, 1.0, 1.0 / 0.03204])
    x = n / np.sum(n)
    rho = pcsaft_den(t, p, x, params, phase='liq')

    terms = pcsaft_lnfugcoef_terms(t, rho, x, params)
    expected = {
        'mu_hc', 'mu_disp', 'mu_polar', 'mu_assoc', 'mu_ion', 'mu_born', 'mu_total',
        'lnfugcoef_hc', 'lnfugcoef_disp', 'lnfugcoef_polar', 'lnfugcoef_assoc', 'lnfugcoef_ion',
        'lnfugcoef_born', 'lnfugcoef_total', 'lnfugcoef',
        'dadx_hc', 'dadx_disp', 'dadx_polar', 'dadx_assoc', 'dadx_ion', 'dadx_born'
    }
    expected_scalars = {
        'a_hc', 'a_disp', 'a_polar', 'a_assoc', 'a_ion', 'a_born',
        'sum_x_dadx_hc', 'sum_x_dadx_disp', 'sum_x_dadx_polar', 'sum_x_dadx_assoc',
        'sum_x_dadx_ion', 'sum_x_dadx_born',
        'z_raw_hc', 'z_raw_disp', 'z_raw_polar', 'z_raw_assoc', 'z_raw_ion', 'z_raw_born',
        'z_hc', 'z_disp', 'z_polar', 'z_assoc', 'z_ion', 'z_born', 'z_total',
    }
    assert expected.issubset(set(terms.keys()))
    assert expected_scalars.issubset(set(terms.keys()))

    ncomp = len(x)
    for key in expected:
        arr = np.asarray(terms[key], dtype=float)
        assert arr.shape == (ncomp,)
        assert np.all(np.isfinite(arr))
    for key in expected_scalars:
        assert np.isfinite(float(terms[key]))

    mu_sum = (
        np.asarray(terms['mu_hc'])
        + np.asarray(terms['mu_disp'])
        + np.asarray(terms['mu_polar'])
        + np.asarray(terms['mu_assoc'])
        + np.asarray(terms['mu_ion'])
        + np.asarray(terms['mu_born'])
    )
    assert np.allclose(mu_sum, np.asarray(terms['mu_total']), rtol=0.0, atol=1e-12)

    for suffix in ('hc', 'disp', 'polar', 'ion', 'born'):
        recon = (
            float(terms[f'a_{suffix}'])
            + float(terms[f'z_raw_{suffix}'])
            + np.asarray(terms[f'dadx_{suffix}'], dtype=float)
            - float(terms[f'sum_x_dadx_{suffix}'])
        )
        assert np.allclose(recon, np.asarray(terms[f'mu_{suffix}']), rtol=0.0, atol=1e-12)

    lnfug_sum = (
        np.asarray(terms['lnfugcoef_hc'])
        + np.asarray(terms['lnfugcoef_disp'])
        + np.asarray(terms['lnfugcoef_polar'])
        + np.asarray(terms['lnfugcoef_assoc'])
        + np.asarray(terms['lnfugcoef_ion'])
        + np.asarray(terms['lnfugcoef_born'])
    )
    assert np.allclose(lnfug_sum, np.asarray(terms['lnfugcoef_total']), rtol=0.0, atol=1e-12)

    z_sum = (
        float(terms['z_hc'])
        + float(terms['z_disp'])
        + float(terms['z_polar'])
        + float(terms['z_assoc'])
        + float(terms['z_ion'])
        + float(terms['z_born'])
    )
    assert abs(z_sum - (float(terms['z_total']) - 1.0)) < 1e-12


def test_lnfugcoef_terms_near_ideal_stable():
    t = 430.0
    p = 10.0
    x = np.asarray([1.0])
    params = {
        'm': np.asarray([2.0020]),
        's': np.asarray([3.6184]),
        'e': np.asarray([208.11]),
    }

    rho = pcsaft_den(t, p, x, params, phase='vap')
    terms = pcsaft_lnfugcoef_terms(t, rho, x, params)

    assert abs(float(terms['z_total']) - 1.0) < 1e-4
    for key in (
        'lnfugcoef_hc', 'lnfugcoef_disp', 'lnfugcoef_polar',
        'lnfugcoef_assoc', 'lnfugcoef_ion', 'lnfugcoef_born', 'lnfugcoef_total'
    ):
        arr = np.asarray(terms[key], dtype=float)
        assert np.all(np.isfinite(arr))

    lnfug_sum = (
        np.asarray(terms['lnfugcoef_hc'])
        + np.asarray(terms['lnfugcoef_disp'])
        + np.asarray(terms['lnfugcoef_polar'])
        + np.asarray(terms['lnfugcoef_assoc'])
        + np.asarray(terms['lnfugcoef_ion'])
        + np.asarray(terms['lnfugcoef_born'])
    )
    assert np.allclose(lnfug_sum, np.asarray(terms['lnfugcoef_total']), rtol=0.0, atol=1e-12)


def test_gsolv(print_result=False):
    """Test ion-wise infinite-dilution Gibbs solvation energy output format."""
    t = 298.15  # K
    p = 101325
    species = ['Na+', 'Cl-', 'H2O-2B-Li']
    ref1, ref2 = -378378.3784, -312883.4356

    canonical = json.loads(
        _dataset_file("2025_Figiel", "user_options.json").read_text(encoding="utf-8")
    )
    runtime = _resolve_runtime_options(canonical)["runtime"]
    runtime["dielc_diff_mode"] = 0
    runtime["debug"] = True

    params = {
        "MW": np.asarray([22.98e-3, 35.45e-3, 18.01528e-3]),
        "m": np.asarray([1.0, 1.0, 1.2047]),
        "s": np.asarray([2.8232, 2.7560, 2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)]),
        "e": np.asarray([230.0, 170.0, 353.95]),
        "e_assoc": np.asarray([0.0, 0.0, 2425.7]),
        "vol_a": np.asarray([0.0, 0.0, 0.04509]),
        "assoc_scheme": [None, None, "2B"],
        "dipm": np.asarray([0.0, 0.0, 0.0]),
        "dip_num": np.asarray([1.0, 1.0, 1.0]),
        "z": np.asarray([1.0, -1.0, 0.0]),
        "dielc": np.asarray([8.0, 8.0, 78.09]),
        "d_born": np.asarray([3.445, 4.1, 0.0]),
        "f_solv": np.asarray([1.0, 1.0, 1.5]),
        "k_ij": np.asarray([
            [0.0, 0.8, -0.3],
            [0.8, 0.0, -0.3],
            [-0.3, -0.3, 0.0],
        ]),
        "l_ij": np.zeros((3, 3)),
        "k_hb": np.zeros((3, 3)),
        "elec_model": _runtime_to_elec_model(runtime),
        "debug": bool(runtime["debug"]),
    }

    n = np.asarray([1.0, 1.0, 1.0 / 0.01801528])
    x = n / np.sum(n)
    rho_model = pcsaft_den(t, p, x, params, phase='liq')

    calc_model = pcsaft_gsolv(t, rho_model, x, params, species=species)
    calc1, calc2 = calc_model.values()
    if print_result:
        print('##########  Test with NaCl  ##########')
        print('----- Gibbs Energy of Solvation in Water at 298.15 K -----')
        print('----- Na+ -------')
        print('    Reference:', ref1, 'J mol^-1')
        print('    PC-SAFT:', calc1, 'J mol^-1')
        print('    Relative deviation:', (calc1 - ref1) / ref1 * 100, '%')
        print('----- Cl- -------')
        print('    Reference:', ref2, 'J mol^-1')
        print('    PC-SAFT:', calc2, 'J mol^-1')
        print('    Relative deviation:', (calc2 - ref2) / ref2 * 100, '%')
    assert set(calc_model.keys()) == {'Na+', 'Cl-'}
    assert abs((calc1 - ref1) / ref1 * 100) < 10
    assert abs((calc2 - ref2) / ref2 * 100) < 10


def test_Hvap(print_result=False):
    """Test the enthalpy of vaporization function to see if it is working correctly."""
    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    params = {'m': m, 's': s, 'e': e}

    ref = 33500.  # source: DIPPR correlation
    p = 90998.  # source: reference equation of state from Polt, A.; Platzer, B.; Maurer, G., Parameter der thermischen Zustandsgleichung von Bender fuer 14 mehratomige reine Stoffe, Chem. Tech. (Leipzig), 1992, 44, 6, 216-224.
    calc = pcsaft_Hvap(380., x, params)[0]
    if print_result:
        print('##########  Test with toluene  ##########')
        print('----- Enthalpy of vaporization at 380 K -----')
        print('    Reference:', ref, 'J mol^-1')
        print('    PC-SAFT:', calc, 'J mol^-1')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    # Water
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    ref = 44761.23  # source: IAWPS95 EOS
    p = 991.82  # source: IAWPS95 EOS
    t = 280
    s = np.asarray([2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}
    calc = pcsaft_Hvap(t, x, params)[0]
    if print_result:
        print('\n##########  Test with water  ##########')
        print('----- Enthalpy of vaporization at 280 K -----')
        print('    Reference:', ref, 'J mol^-1')
        print('    PC-SAFT:', calc, 'J mol^-1')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    # Dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    params = {'m': m, 's': s, 'e': e, 'dipm': dpm, 'dip_num': dip_num}

    ref = 17410.  # source: DIPPR correlation
    p = 937300.  # source: DIPPR correlation
    calc = pcsaft_Hvap(315., x, params)[0]
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('----- Enthalpy of vaporization at 315 K -----')
        print('    Reference:', ref, 'J mol^-1')
        print('    PC-SAFT:', calc, 'J mol^-1')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

def test_dadt(print_result=False):
    """Test the function for the temperature derivative of the Helmholtz energy."""
    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    params = {'m': m, 's': s, 'e': e}

    p = 100000.
    t = 330.

    rho = pcsaft_den(t, p, x, params, phase='liq')
    dadt_eos = pcsaft_dadt(t, rho, x, params)

    # calculating numerical derivative
    der1 = pcsaft_ares(t - 1, rho, x, params)
    der2 = pcsaft_ares(t + 1, rho, x, params)
    dadt_num = (der2 - der1) / 2.
    if print_result:
        print('\n##########  Test with toluene  ##########')
        print('    Numerical derivative:', dadt_num)
        print('    PC-SAFT derivative:', dadt_eos)
        print('    Relative deviation:', (dadt_eos - dadt_num) / dadt_num * 100, '%')
    assert abs((dadt_eos - dadt_num) / dadt_num * 100) < 2e-2

    # Acetic acid
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}

    p = 100000.
    t = 310.

    rho = pcsaft_den(t, p, x, params, phase='liq')
    dadt_eos = pcsaft_dadt(t, rho, x, params)

    # calculating numerical derivative
    der1 = pcsaft_ares(t - 1, rho, x, params)
    der2 = pcsaft_ares(t + 1, rho, x, params)
    dadt_num = (der2 - der1) / 2.
    if print_result:
        print('\n##########  Test with acetic acid  ##########')
        print('    Numerical derivative:', dadt_num)
        print('    PC-SAFT derivative:', dadt_eos)
        print('    Relative deviation:', (dadt_eos - dadt_num) / dadt_num * 100, '%')
    assert abs((dadt_eos - dadt_num) / dadt_num * 100) < 2e-2

    # Water
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    p = 100000.
    t = 290.

    s = np.asarray([2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}

    rho = pcsaft_den(t, p, x, params, phase='liq')
    dadt_eos = pcsaft_dadt(t, rho, x, params)

    # calculating numerical derivative
    der1 = pcsaft_ares(t - 1, rho, x, params)
    der2 = pcsaft_ares(t + 1, rho, x, params)
    dadt_num = (der2 - der1) / 2.
    if print_result:
        print('\n##########  Test with water  ##########')
        print('    Numerical derivative:', dadt_num)
        print('    PC-SAFT derivative:', dadt_eos)
        print('    Relative deviation:', (dadt_eos - dadt_num) / dadt_num * 100, '%')
    assert abs((dadt_eos - dadt_num) / dadt_num * 100) < 2e-2

    # Dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    params = {'m': m, 's': s, 'e': e, 'dipm': dpm, 'dip_num': dip_num}

    p = 100000.
    t = 370.

    rho = pcsaft_den(t, p, x, params, phase='liq')
    dadt_eos = pcsaft_dadt(t, rho, x, params)

    # calculating numerical derivative
    der1 = pcsaft_ares(t - 1, rho, x, params)
    der2 = pcsaft_ares(t + 1, rho, x, params)
    dadt_num = (der2 - der1) / 2.
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('    Numerical derivative:', dadt_num)
        print('    PC-SAFT derivative:', dadt_eos)
        print('    Relative deviation:', (dadt_eos - dadt_num) / dadt_num * 100, '%')
    assert abs((dadt_eos - dadt_num) / dadt_num * 100) < 2e-2

    # Aqueous NaCl
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.0907304774758426, 0.0907304774758426, 0.818539045048315])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                       [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    t = 298.15  # K
    p = 100000.  # Pa
    s[2] = 2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)  # temperature dependent segment diameter for water
    k_ij[0, 2] = -0.007981 * t + 2.37999
    k_ij[2, 0] = -0.007981 * t + 2.37999
    dielc = dielc_water(t)

    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij, 'z': z, 'dielc': np.full(len(m), dielc)}

    rho = pcsaft_den(t, p, x, params, phase='liq')
    dadt_eos = pcsaft_dadt(t, rho, x, params)

    # calculating numerical derivative
    der1 = pcsaft_ares(t - 1, rho, x, params)
    der2 = pcsaft_ares(t + 1, rho, x, params)
    dadt_num = (der2 - der1) / 2.
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('    Numerical derivative:', dadt_num)
        print('    PC-SAFT derivative:', dadt_eos)
        print('    Relative deviation:', (dadt_eos - dadt_num) / dadt_num * 100, '%')
    assert abs((dadt_eos - dadt_num) / dadt_num * 100) < 2e-2

def test_cp(print_result=False):
    """Test the heat capacity function to see if it is working correctly."""
    # Benzene
    x = np.asarray([1.])
    m = np.asarray([2.4653])
    s = np.asarray([3.6478])
    e = np.asarray([287.35])
    cnsts = np.asarray([55238., 173380, 764.25, 72545, 2445.7])  # constants for Aly-Lee equation (obtained from DIPPR)
    params = {'m': m, 's': s, 'e': e}

    ref = 140.78  # source: Equation of state from Polt et al. (1992) (available at https://webbook.nist.gov/chemistry/fluid/)
    p = 100000.
    t = 330.
    rho = pcsaft_den(t, p, x, params, phase='liq')
    calc = pcsaft_cp(t, rho, cnsts, x, params)
    if print_result:
        print('\n##########  Test with benzene  ##########')
        print('----- Heat capacity at 330 K -----')
        print('    Reference:', ref, 'J mol^-1 K^-1')
        print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    # Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    cnsts = np.asarray([58140., 286300, 1440.6, 189800, 650.43])  # constants for Aly-Lee equation (obtained from DIPPR)
    params = {'m': m, 's': s, 'e': e}

    ref = 179.79  # source: Equation of state from Polt et al. (1992) (available at https://webbook.nist.gov/chemistry/fluid/)
    p = 100000.
    t = 370.
    rho = pcsaft_den(t, p, x, params, phase='liq')
    calc = pcsaft_cp(t, rho, cnsts, x, params)
    if print_result:
        print('\n##########  Test with toluene  ##########')
        print('----- Heat capacity at 370 K -----')
        print('    Reference:', ref, 'J mol^-1 K^-1')
        print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

    # # Acetic acid
    # # print('\n##########  Test with acetic acid  ##########')
    # m = np.asarray([1.3403])
    # s = np.asarray([3.8582])
    # e = np.asarray([211.59])
    # volAB = np.asarray([0.075550])
    # eAB = np.asarray([3044.4])
    # cnsts = np.asarray([40200., 136750, 1262, 70030, 569.7]) # constants for Aly-Lee equation (obtained from DIPPR)
    # params = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB}
    #
    # ref = 130.3 # source: DIPPR
    # p = 100000.
    # t = 325.
    # rho = pcsaft_den(t, p, x, params, phase='liq')
    # calc = pcsaft_cp(t, rho, cnsts, x, params)
    # """ Note: Large deviations occur with acetic acid and water. This behavior
    # has been observed before and was described in R. T. C. S. Ribeiro, A. L.
    # Alberton, M. L. L. Paredes, G. M. Kontogeorgis, and X. Liang, “Extensive
    # Study of the Capabilities and Limitations of the CPA and sPC-SAFT Equations
    # of State in Modeling a Wide Range of Acetic Acid Properties,” Ind. Eng.
    # Chem. Res., vol. 57, no. 16, pp. 5690–5704, Apr. 2018. """
    # # print('----- Heat capacity at 325 K -----')
    # # print('    Reference:', ref, 'J mol^-1 K^-1')
    # # print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
    # # print('    Relative deviation:', (calc-ref)/ref*100, '%')

    # Dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    cnsts = np.asarray([57431., 94494, 895.51, 65065, 2467.4])  # constants for Aly-Lee equation (obtained from DIPPR)
    params = {'m': m, 's': s, 'e': e, 'dipm': dpm, 'dip_num': dip_num}

    ref = 102.2  # source: DIPPR correlation
    p = 100000.
    t = 240.
    rho = pcsaft_den(t, p, x, params, phase='liq')
    calc = pcsaft_cp(t, rho, cnsts, x, params)
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('----- Heat capacity at 240 K -----')
        print('    Reference:', ref, 'J mol^-1 K^-1')
        print('    PC-SAFT:', calc, 'J mol^-1 K^-1')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 3

def test_pressure(print_result=False):
    """Test the pressure function to see if it is working correctly."""
    #     Toluene
    x = np.asarray([1.])
    m = np.asarray([2.8149])
    s = np.asarray([3.7169])
    e = np.asarray([285.69])
    params = {'m': m, 's': s, 'e': e}

    ref = 101325  # Pa
    t = 320  # K
    rho = 9033.11421467559  # mol m^-3 From density calculation with working PC-SAFT density function
    calc = pcsaft_p(t, rho, x, params)
    if print_result:
        print('\n##########  Test with toluene  ##########')
        print('----- Pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 1e-6

    # Water
    m = np.asarray([1.2047])
    e = np.asarray([353.95])
    volAB = np.asarray([0.0451])
    eAB = np.asarray([2425.67])

    ref = 101325  # Pa
    t = 274  # K
    s = np.asarray([2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}
    rho = 55476.442745328946  # mol m^-3 From density calculation with working PC-SAFT density function
    calc = pcsaft_p(t, rho, x, params)
    if print_result:
        print('\n##########  Test with water  ##########')
        print('----- Pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 1e-6

    # Acetic acid
    m = np.asarray([1.3403])
    s = np.asarray([3.8582])
    e = np.asarray([211.59])
    volAB = np.asarray([0.075550])
    eAB = np.asarray([3044.4])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB}

    ref = 101325  # Pa
    t = 305  # K
    rho = 16965.669448881614  # mol m^-3 From density calculation with working PC-SAFT density function
    calc = pcsaft_p(t, rho, x, params)
    if print_result:
        print('\n##########  Test with acetic acid  ##########')
        print('----- Pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 1e-6

    # dimethyl ether
    m = np.asarray([2.2634])
    s = np.asarray([3.2723])
    e = np.asarray([210.29])
    dpm = np.asarray([1.3])
    dip_num = np.asarray([1.0])
    params = {'m': m, 's': s, 'e': e, 'dipm': dpm, 'dip_num': dip_num}

    ref = 101325  # Pa
    t = 240  # K
    rho = 15955.509156351702  # mol m^-3 From density calculation with working PC-SAFT density function
    calc = pcsaft_p(t, rho, x, params)
    if print_result:
        print('\n##########  Test with dimethyl ether  ##########')
        print('----- Pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 1e-6

    # Binary mixture: methanol-cyclohexane
    #0 = methanol, 1 = cyclohexane
    x = np.asarray([0.0550, 0.945])
    m = np.asarray([1.5255, 2.5303])
    s = np.asarray([3.2300, 3.8499])
    e = np.asarray([188.90, 278.11])
    volAB = np.asarray([0.035176, 0.])
    eAB = np.asarray([2899.5, 0.])
    k_ij = np.asarray([[0, 0.051],
                       [0.051, 0]])
    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij}

    ref = 101325  # Pa
    t = 298.15  # K
    rho = 9368.903688400374  # mol m^-3 From density calculation with working PC-SAFT density function
    calc = pcsaft_p(t, rho, x, params)
    if print_result:
        print('\n##########  Test with methanol-cyclohexane mixture  ##########')
        print('----- Pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 1e-6

    # # Binary mixture: hydrogen-toluene
    # # Although the results from this code match the values from Joachim Gross's code, neither code
    # # give a pressure that matches literature values. Further investigation is needed to figure out why.
    # # 0 = hydrogen, 1 = toluene
    # x0 = 0.037
    # x = np.asarray([x0, 1-x0])
    # m = np.asarray([1.0000, 2.8149])
    # s = np.asarray([2.9860, 3.7169])
    # e = np.asarray([19.2775, 285.69])
    # k_ij = np.asarray([[0, 0],
    #                    [0, 0]])
    # params = {'m':m, 's':s, 'e':e, 'k_ij':k_ij}
    #
    # t = 501.6 # K
    # ref = 50.0 * 1e5 # Pa, source: Lin, H.-M.; Sebastian,H.M.; Chao,K.-C.; J. Chem. Engng. Data. 1980, 25, 252-257.
    # rho = 7090.18885 # mol m^-3
    # calc = pcsaft_p(t, rho, x, params)
    # if print_result:
    #     print('\n##########  Test with hydrogen-toluene mixture  ##########')
    #     print('----- Pressure at %s K -----' % t)
    #     print('    Reference:', ref, 'Pa')
    #     print('    PC-SAFT:', calc, 'Pa')
    #     print('    Relative deviation:', (calc-ref)/ref*100, '%')
    # assert abs((calc-ref)/ref*100) < 10

    # Binary mixture: water-acetic acid
    #0 = water, 1 = acetic acid
    x = np.asarray([0.9898662364, 0.0101337636])
    m = np.asarray([1.2047, 1.3403])
    s = np.asarray([0, 3.8582])
    e = np.asarray([353.95, 211.59])
    volAB = np.asarray([0.0451, 0.075550])
    eAB = np.asarray([2425.67, 3044.4])
    k_ij = np.asarray([[0, -0.127],
                       [-0.127, 0]])

    t = 403.574
    s[0] = 3.8395 + 1.2828 * np.exp(-0.0074944 * t) - 1.3939 * np.exp(-0.00056029 * t)
    params = {'x': x, 'm': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij}
    ref = 275000.  # experimental bubble point pressure is 273722 Pa. source: Othmer, D. F.; Silvis, S. J.; Spiel, A. Ind. Eng. Chem., 1952, 44, 1864-72 Composition of vapors from boiling binary solutions pressure equilibrium still for studying water - acetic acid system
    xv_ref = np.asarray([0.9923666645, 0.0076333355])
    rho = 50902.74844165996
    calc = pcsaft_p(t, rho, x, params)
    if print_result:
        print('\n##########  Test with water-acetic acid mixture  ##########')
        print('----- Bubble point pressure at %s K -----' % t)
        print('    Liquid composition:', x)
        print('    Reference pressure:', ref, 'Pa')
        print('    PC-SAFT pressure:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 1e-6

    # NaCl in water
    # 0 = Na+, 1 = Cl-, 2 = H2O
    x = np.asarray([0.010579869455908, 0.010579869455908, 0.978840261088184])
    m = np.asarray([1, 1, 1.2047])
    s = np.asarray([2.8232, 2.7599589, 0.])
    e = np.asarray([230.00, 170.00, 353.9449])
    volAB = np.asarray([0, 0, 0.0451])
    eAB = np.asarray([0, 0, 2425.67])
    k_ij = np.asarray([[0, 0.317, 0],
                       [0.317, 0, -0.25],
                       [0, -0.25, 0]])
    z = np.asarray([1., -1., 0.])

    ref = 101325  # Pa
    t = 298.15  # K
    s[2] = 2.7927 + 10.11 * np.exp(-0.01775 * t) - 1.417 * np.exp(-0.01146 * t)  # temperature dependent segment diameter for water
    k_ij[0, 2] = -0.007981 * t + 2.37999
    k_ij[2, 0] = -0.007981 * t + 2.37999
    dielc = dielc_water(t)
    rho = 55757.07260200306  # mol m^-3 From density calculation with working PC-SAFT density function

    params = {'m': m, 's': s, 'e': e, 'e_assoc': eAB, 'vol_a': volAB, 'k_ij': k_ij, 'z': z, 'dielc': np.full(len(m), dielc)}

    calc = pcsaft_p(t, rho, x, params)
    if print_result:
        print('\n##########  Test with aqueous NaCl  ##########')
        print('----- Pressure at {} K -----'.format(t))
        print('    Reference:', ref, 'Pa')
        print('    PC-SAFT:', calc, 'Pa')
        print('    Relative deviation:', (calc - ref) / ref * 100, '%')
    assert abs((calc - ref) / ref * 100) < 1e-6

if __name__ == '__main__':
    # test_ares(print_result=True)
    test_gsolv(print_result=True)
    # test_miac_m(print_result=True)
    # test_osmoticC(print_result=True)


def _load_dataset_params(dataset, species, x, t, user_options=None):
    return get_prop_dict(dataset, species, np.asarray(x, dtype=float), t, user_options=user_options or {})


def test_resolve_runtime_mu_dh_defaults():
    canonical = json.loads(
        _dataset_file("2014_Held", "user_options.json").read_text(encoding="utf-8")
    )
    runtime = _resolve_runtime_options(canonical)["runtime"]
    assert runtime["mu_DH_diff_mode"] == 0
    assert runtime["mu_DH_comp_dep_rel_perm"] is True
    assert runtime["mu_DH_include_sum_term"] is True
    assert runtime["hc_dadx_diff_mode"] == 0
    assert runtime["disp_dadx_diff_mode"] == 0
    assert runtime["assoc_dadx_diff_mode"] == 0
    assert runtime["polar_dadx_diff_mode"] == 0


def test_create_struct_rejects_flat_mu_dh_keys():
    t = 298.15
    p = 1.0e5
    x = np.asarray([0.02, 0.02, 0.96])
    params = _load_dataset_params("2020_Bulow", ["Li+", "Br-", "Ethanol"], x, t)
    params["mu_DH_diff_mode"] = 1
    with pytest.raises(ValueError, match='Flat electrostatic params are no longer supported'):
        rho = pcsaft_den(t, p, x, params, phase='liq')
        pcsaft_lnfugcoef_terms(t, rho, x, params)


@pytest.mark.parametrize(
    "dataset,species,x,t",
    [
        ("2020_Bulow", ["Li+", "Br-", "Ethanol"], np.asarray([0.03, 0.03, 0.94]), 298.15),
        ("2025_Figiel", ["Na+", "Cl-", "H2O-2B-Li"], np.asarray([0.02, 0.02, 0.96]), 298.15),
    ],
)
def test_mu_dh_analytical_numeric_close(dataset, species, x, t):
    analytical = _load_dataset_params(
        dataset,
        species,
        x,
        t,
        user_options={"elec_model": {"DH_model": {"mu_DH_model": {"differential_mode": "analytical"}}}},
    )
    numerical = _load_dataset_params(
        dataset,
        species,
        x,
        t,
        user_options={"elec_model": {"DH_model": {"mu_DH_model": {"differential_mode": "numerical"}}}},
    )
    rho = pcsaft_den(t, 1.0e5, x, analytical, phase='liq')
    mu_dh_analytical = np.asarray(pcsaft_lnfugcoef_terms(t, rho, x, analytical)["mu_ion"], dtype=float)
    mu_dh_numerical = np.asarray(pcsaft_lnfugcoef_terms(t, rho, x, numerical)["mu_ion"], dtype=float)
    assert np.allclose(mu_dh_analytical, mu_dh_numerical, rtol=0.0, atol=2e-5)


@pytest.mark.parametrize(
    "dataset,species,x,t",
    [
        ("2020_Bulow", ["Li+", "Br-", "Ethanol"], np.asarray([0.03, 0.03, 0.94]), 298.15),
        ("2025_Figiel", ["Na+", "Cl-", "H2O-2B-Li"], np.asarray([0.02, 0.02, 0.96]), 298.15),
    ],
)
def test_mu_dh_analytical_autodiff_close(dataset, species, x, t):
    analytical = _load_dataset_params(
        dataset,
        species,
        x,
        t,
        user_options={"elec_model": {"DH_model": {"mu_DH_model": {"differential_mode": "analytical"}}}},
    )
    autodiff = _load_dataset_params(
        dataset,
        species,
        x,
        t,
        user_options={"elec_model": {"DH_model": {"mu_DH_model": {"differential_mode": "autodiff"}}}},
    )
    rho = pcsaft_den(t, 1.0e5, x, analytical, phase='liq')
    mu_dh_analytical = np.asarray(pcsaft_lnfugcoef_terms(t, rho, x, analytical)["mu_ion"], dtype=float)
    mu_dh_autodiff = np.asarray(pcsaft_lnfugcoef_terms(t, rho, x, autodiff)["mu_ion"], dtype=float)
    assert np.allclose(mu_dh_analytical, mu_dh_autodiff, rtol=0.0, atol=5e-10)


def test_mu_dh_toggle_changes_only_dh_branch():
    t = 298.15
    x = np.asarray([0.03, 0.03, 0.94])
    species = ["Li+", "Br-", "Ethanol"]
    base = _load_dataset_params("2020_Bulow", species, x, t)
    no_deps = _load_dataset_params(
        "2020_Bulow",
        species,
        x,
        t,
        user_options={"elec_model": {"DH_model": {"mu_DH_model": {"comp_dep_rel_perm": False}}}},
    )
    no_sum = _load_dataset_params(
        "2020_Bulow",
        species,
        x,
        t,
        user_options={"elec_model": {"DH_model": {"mu_DH_model": {"include_sum_term": False}}}},
    )
    rho = pcsaft_den(t, 1.0e5, x, base, phase='liq')
    terms_base = pcsaft_lnfugcoef_terms(t, rho, x, base)
    terms_no_deps = pcsaft_lnfugcoef_terms(t, rho, x, no_deps)
    terms_no_sum = pcsaft_lnfugcoef_terms(t, rho, x, no_sum)

    mu_base = np.asarray(terms_base["mu_ion"], dtype=float)
    mu_no_deps = np.asarray(terms_no_deps["mu_ion"], dtype=float)
    mu_no_sum = np.asarray(terms_no_sum["mu_ion"], dtype=float)
    assert np.max(np.abs(mu_no_deps - mu_base)) > 1e-8
    assert np.max(np.abs(mu_no_sum - mu_base)) > 1e-8
    assert np.allclose(np.asarray(terms_base["mu_assoc"], dtype=float), np.asarray(terms_no_deps["mu_assoc"], dtype=float), rtol=0.0, atol=1e-12)
    assert np.allclose(np.asarray(terms_base["mu_assoc"], dtype=float), np.asarray(terms_no_sum["mu_assoc"], dtype=float), rtol=0.0, atol=1e-12)


def test_contribution_dadx_numeric_modes_are_available():
    t = 298.15
    x = np.asarray([0.03, 0.03, 0.94])
    species = ["Li+", "Br-", "Ethanol"]
    analytical = _load_dataset_params("2020_Bulow", species, x, t)
    numerical = _load_dataset_params(
        "2020_Bulow",
        species,
        x,
        t,
        user_options={
            "elec_model": {
                "hc_model": {"dadx_differential_mode": "numerical"},
                "disp_model": {"dadx_differential_mode": "numerical"},
                "assoc_model": {"dadx_differential_mode": "numerical"},
                "polar_model": {"dadx_differential_mode": "numerical"},
            }
        },
    )
    rho = pcsaft_den(t, 1.0e5, x, analytical, phase='liq')
    terms_analytical = pcsaft_lnfugcoef_terms(t, rho, x, analytical)
    terms_numerical = pcsaft_lnfugcoef_terms(t, rho, x, numerical)
    for key in ("mu_hc", "mu_disp", "mu_assoc", "mu_polar"):
        arr_a = np.asarray(terms_analytical[key], dtype=float)
        arr_n = np.asarray(terms_numerical[key], dtype=float)
        assert np.all(np.isfinite(arr_a))
        assert np.all(np.isfinite(arr_n))
    assert np.max(
        np.abs(np.asarray(terms_numerical["mu_disp"], dtype=float) - np.asarray(terms_analytical["mu_disp"], dtype=float))
    ) > 1e-10


def test_dielc_analytical_autodiff_close():
    t = 298.15
    x = np.asarray([0.03, 0.03, 0.94])
    species = ["Li+", "Br-", "Ethanol"]
    analytical = _load_dataset_params(
        "2020_Bulow",
        species,
        x,
        t,
        user_options={"elec_model": {"rel_perm": {"differential_mode": "analytical"}}},
    )
    autodiff = _load_dataset_params(
        "2020_Bulow",
        species,
        x,
        t,
        user_options={"elec_model": {"rel_perm": {"differential_mode": "autodiff"}}},
    )
    eps_analytical, deps_analytical = pcsaft_dielc_eval(x, analytical)
    eps_autodiff, deps_autodiff = pcsaft_dielc_eval(x, autodiff)
    assert eps_analytical == pytest.approx(eps_autodiff, rel=0.0, abs=1.0e-12)
    assert np.allclose(np.asarray(deps_analytical, dtype=float), np.asarray(deps_autodiff, dtype=float), rtol=0.0, atol=5e-12)


@pytest.mark.parametrize(
    "dataset,species,x,t,baseline_mode,atol",
    [
        ("2020_Bulow", ["Li+", "Br-", "Ethanol"], np.asarray([0.03, 0.03, 0.94]), 298.15, "analytical", 5e-10),
        ("2025_Figiel", ["Na+", "Cl-", "H2O-2B-Li"], np.asarray([0.02, 0.02, 0.96]), 298.15, "numerical", 5e-10),
    ],
)
def test_mu_born_analytical_autodiff_close(dataset, species, x, t, baseline_mode, atol):
    baseline = _load_dataset_params(
        dataset,
        species,
        x,
        t,
        user_options={"elec_model": {"born_model": {"mu_born_model": {"differential_mode": baseline_mode}}}},
    )
    autodiff = _load_dataset_params(
        dataset,
        species,
        x,
        t,
        user_options={"elec_model": {"born_model": {"mu_born_model": {"differential_mode": "autodiff"}}}},
    )
    rho = pcsaft_den(t, 1.0e5, x, baseline, phase='liq')
    mu_born_baseline = np.asarray(pcsaft_lnfugcoef_terms(t, rho, x, baseline)["mu_born"], dtype=float)
    mu_born_autodiff = np.asarray(pcsaft_lnfugcoef_terms(t, rho, x, autodiff)["mu_born"], dtype=float)
    assert np.allclose(mu_born_baseline, mu_born_autodiff, rtol=0.0, atol=atol)


def test_contribution_dadx_autodiff_matches_analytical():
    t = 298.15
    x = np.asarray([0.03, 0.03, 0.94])
    species = ["Li+", "Br-", "Ethanol"]
    analytical = _load_dataset_params("2020_Bulow", species, x, t)
    autodiff = _load_dataset_params(
        "2020_Bulow",
        species,
        x,
        t,
        user_options={
            "elec_model": {
                "hc_model": {"dadx_differential_mode": "autodiff"},
                "disp_model": {"dadx_differential_mode": "autodiff"},
                "assoc_model": {"dadx_differential_mode": "autodiff"},
                "polar_model": {"dadx_differential_mode": "autodiff"},
            }
        },
    )
    rho = pcsaft_den(t, 1.0e5, x, analytical, phase='liq')
    terms_analytical = pcsaft_lnfugcoef_terms(t, rho, x, analytical)
    terms_autodiff = pcsaft_lnfugcoef_terms(t, rho, x, autodiff)
    for key in ("mu_hc", "mu_disp", "mu_assoc", "mu_polar"):
        arr_a = np.asarray(terms_analytical[key], dtype=float)
        arr_ad = np.asarray(terms_autodiff[key], dtype=float)
        assert np.all(np.isfinite(arr_a))
        assert np.all(np.isfinite(arr_ad))
        assert np.allclose(arr_a, arr_ad, rtol=0.0, atol=5e-10)


def test_2020_Bulow_ethanol_transfer_contribution_sum_matches_total():
    t = 298.15
    p = 1.0e5
    eps = 1e-8
    eps_inf = 1e-12
    r_gas = 8.31446261815324

    def species_for_ion(ion):
        if ion in {"Li+", "Na+", "K+"}:
            return [ion, "Cl-", "Ethanol"]
        return ["Na+", ion, "Ethanol"]

    def lnfug_terms_for_state(species, ion):
        x = np.asarray([eps, eps, 1.0 - 2.0 * eps], dtype=float)
        params = get_prop_dict("2020_Bulow", species, x, t, user_options={})
        rho = pcsaft_den(t, p, x, params, phase='liq')
        z = np.asarray(params["z"], dtype=float)
        idx_ion = np.where(np.abs(z) > 1.0e-12)[0]
        idx_solv = np.where(np.abs(z) <= 1.0e-12)[0]
        x_ref = x.copy()
        x_ref[idx_ion] = 0.0
        x_ref[idx_solv] = x_ref[idx_solv] / np.sum(x_ref[idx_solv])
        p_ref = pcsaft_p(t, rho, x_ref, params)
        x_inf = x_ref.copy()
        ion_idx = species.index(ion)
        x_inf[ion_idx] = eps_inf
        x_inf /= np.sum(x_inf)
        rho_inf = pcsaft_den(t, p_ref, x_inf, params, phase='liq')
        terms = pcsaft_lnfugcoef_terms(t, rho_inf, x_inf, params)
        values = pcsaft_gsolv(t, rho, x, params, species=species)
        return terms, values, ion_idx

    contribution_keys = (
        "lnfugcoef_hc",
        "lnfugcoef_disp",
        "lnfugcoef_assoc",
        "lnfugcoef_ion",
        "lnfugcoef_born",
    )

    for ion in ("Na+", "Cl-", "I-"):
        species_ethanol = species_for_ion(ion)
        terms_ethanol, gsolv_ethanol, idx_ethanol = lnfug_terms_for_state(species_ethanol, ion)

        species_water = species_ethanol[:-1] + ["Water"]
        terms_water, gsolv_water, idx_water = lnfug_terms_for_state(species_water, ion)

        contribution_sum = 0.0
        for key in contribution_keys:
            contribution_sum += float(
                r_gas * t * (terms_ethanol[key][idx_ethanol] - terms_water[key][idx_water]) / 1000.0
            )

        total_transfer = float(gsolv_ethanol[ion] - gsolv_water[ion]) / 1000.0
        assert abs(contribution_sum - total_transfer) < 1e-10






@pytest.mark.parametrize(
    "dataset,species,x,t,baseline_mode,atol",
    [
        ("2020_Bulow", ["Li+", "Br-", "Ethanol"], np.asarray([0.03, 0.03, 0.94]), 298.15, "analytical", 2e-6),
        ("2025_Figiel", ["Na+", "Cl-", "H2O-2B-Li"], np.asarray([0.02, 0.02, 0.96]), 298.15, "numerical", 2e-6),
    ],
)
def test_dadt_mode_selector_matches_expected_baseline(dataset, species, x, t, baseline_mode, atol):
    baseline = _load_dataset_params(dataset, species, x, t, user_options={"dadt_differential_mode": baseline_mode})
    numerical = _load_dataset_params(dataset, species, x, t, user_options={"dadt_differential_mode": "numerical"})
    autodiff = _load_dataset_params(dataset, species, x, t, user_options={"dadt_differential_mode": "autodiff"})
    rho = pcsaft_den(t, 1.0e5, x, baseline, phase='liq')
    dadt_fd = _central_dadt(t, rho, x, baseline)
    dadt_numerical = pcsaft_dadt(t, rho, x, numerical)
    dadt_autodiff = pcsaft_dadt(t, rho, x, autodiff)
    assert dadt_numerical == pytest.approx(dadt_fd, rel=0.0, abs=1e-12)
    assert dadt_autodiff == pytest.approx(dadt_fd, rel=0.0, abs=atol)
    if baseline_mode == "analytical":
        dadt_baseline = pcsaft_dadt(t, rho, x, baseline)
        assert abs((dadt_baseline - dadt_fd) / dadt_fd * 100) < 2e-2


def test_invalid_dadt_differential_mode_raises():
    x = np.asarray([1.0])
    params = {
        'm': np.asarray([2.8149]),
        's': np.asarray([3.7169]),
        'e': np.asarray([285.69]),
        'dadt_differential_mode': 'bad-mode',
    }
    with pytest.raises(ValueError, match='dadt_differential_mode'):
        pcsaft_dadt(330.0, 8000.0, x, params)


@pytest.mark.parametrize(
    "params_builder,x,t,p,phase,temp_h,x_h,rtol,atol",
    [
        (lambda: _neutral_binary_params('autodiff'), np.asarray([0.4, 0.6]), 330.0, 1.0e5, 'liq', 1.0, 1.0e-5, 2e-3, 2e-4),
        (lambda: _load_dataset_params('2020_Bulow', ['Li+', 'Br-', 'Ethanol'], np.asarray([0.03, 0.03, 0.94]), 298.15, user_options={'dadt_differential_mode': 'autodiff'}), np.asarray([0.03, 0.03, 0.94]), 298.15, 1.0e5, 'liq', 1.0, 1.0e-5, 5e-3, 5e-4),
    ],
)
def test_private_autodiff_second_derivatives_match_fd_of_first_derivatives(params_builder, x, t, p, phase, temp_h, x_h, rtol, atol):
    params = params_builder()
    rho = pcsaft_den(t, p, x, params, phase=phase)
    derivs = _pcsaft_autodiff_residual_derivatives(t, rho, x, params)

    d2adt2_fd = (pcsaft_dadt(t + temp_h, rho, x, params) - pcsaft_dadt(t - temp_h, rho, x, params)) / (2.0 * temp_h)
    assert derivs['d2adt2'] == pytest.approx(d2adt2_fd, rel=rtol, abs=atol)

    mixed_fd = np.zeros_like(x, dtype=float)
    hess_fd = np.zeros((len(x), len(x)), dtype=float)
    for j in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[j] += x_h
        xm[j] -= x_h
        mixed_fd[j] = (_pcsaft_autodiff_residual_derivatives(t, rho, xp, params)['dadt'] - _pcsaft_autodiff_residual_derivatives(t, rho, xm, params)['dadt']) / (2.0 * x_h)
        dadx_p = _pcsaft_autodiff_residual_derivatives(t, rho, xp, params)['dadx']
        dadx_m = _pcsaft_autodiff_residual_derivatives(t, rho, xm, params)['dadx']
        hess_fd[:, j] = (dadx_p - dadx_m) / (2.0 * x_h)

    assert np.allclose(np.asarray(derivs['d2adtdx'], dtype=float), mixed_fd, rtol=rtol, atol=atol)
    assert np.allclose(np.asarray(derivs['hessian_x'], dtype=float), hess_fd, rtol=rtol, atol=atol)
