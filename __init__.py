# -*- coding: utf-8 -*-
from . import pcsaft as _pcsaft
from .data.epcsaft_properties import get_prop_dict, validate_species_params

InputError = _pcsaft.InputError
SolutionError = _pcsaft.SolutionError


def _normalize_user_params(user_params):
    if isinstance(user_params, dict):
        return user_params
    return {}

def _parse_salt_name(salt_name):
    if not isinstance(salt_name, str):
        raise InputError('salt name must be a string.')
    plus = salt_name.find('+')
    minus = salt_name.find('-', plus + 1)
    if plus == -1 or minus == -1 or minus < plus:
        raise InputError('salt name must be like "Mg2+Cl-". got: {}'.format(salt_name))
    cation = salt_name[:plus + 1]
    anion = salt_name[plus + 1:]
    if not anion.endswith('-'):
        raise InputError('salt name must end with an anion like "Cl-". got: {}'.format(salt_name))
    return cation, anion

def _resolve_component_spec(species, user_params):
    user_params = _normalize_user_params(user_params)
    roles = user_params.get('component_roles')
    if roles is None:
        return None, None
    names = user_params.get('component_names')
    if names is None:
        if species is not None:
            names = list(species)
        else:
            raise InputError('component_names is required when species is not provided.')
    roles = [str(role).lower() for role in list(roles)]
    names = list(names)
    if len(names) != len(roles):
        raise InputError('component_names and component_roles must have the same length.')
    return names, roles

def _derive_ionic_species_from_components(component_names, component_roles):
    species_list = []
    seen = set()
    for name, role in zip(component_names, component_roles):
        if role in ('solvent', 'neutral'):
            if name not in seen:
                species_list.append(name)
                seen.add(name)
        elif role == 'ion':
            if name not in seen:
                species_list.append(name)
                seen.add(name)
        elif role == 'salt':
            cation, anion = _parse_salt_name(name)
            if cation not in seen:
                species_list.append(cation)
                seen.add(cation)
            if anion not in seen:
                species_list.append(anion)
                seen.add(anion)
    return species_list

def _resolve_species_for_params(species, user_params):
    user_params = _normalize_user_params(user_params)
    comp_names, comp_roles = _resolve_component_spec(species, user_params)
    if comp_roles is not None and any(role == 'salt' for role in comp_roles):
        derived = _derive_ionic_species_from_components(comp_names, comp_roles)
        if species is None:
            return derived
        species_list = list(species)
        if any(name in species_list for name, role in zip(comp_names, comp_roles) if role == 'salt'):
            return derived
        missing = [sp for sp in derived if sp not in species_list]
        if missing:
            raise InputError('species list is missing ions from salts: {}.'.format(', '.join(missing)))
        return species_list
    if species is not None:
        return list(species)
    for key in ('species', 'species_names'):
        if key in user_params:
            return list(user_params[key])
    raise InputError('Composition input requires a species list or user_params["species"].')

def _resolve_params(species, t, user_params, params):
    if params is not None:
        return params
    if species is None:
        raise InputError("Either params or species must be provided.")
    if t is None:
        raise InputError("Temperature must be provided when building params from species.")
    species_params = _resolve_species_for_params(species, user_params)
    params = get_prop_dict(species_params, t, user_params=user_params)
    if isinstance(user_params, dict):
        for key in ('born_model', 'born_enabled', 'bjerrum_model', 'dielc_rule', 'dielc_ion'):
            if key in user_params:
                params[key] = user_params[key]
    return params


def pcsaft_den(t, p, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_den(t, p, x, params, phase=phase, dielc_rule=dielc_rule, species=species, user_params=user_params)


def pcsaft_p(t, rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_p(t, rho, x, params, phase=phase, dielc_rule=dielc_rule, species=species, user_params=user_params)


def pcsaft_Z(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_Z(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_Z_contrib(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_Z_contrib(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_ion_dh_debug(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_ion_dh_debug(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_lnfugcoef(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_lnfugcoef(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_mu_res_contrib(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_mu_res_contrib(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)

def pcsaft_mures(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_mures(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_fugcoef(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_fugcoef(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_lnfugcoef_inf_dil(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p', eps=0.0, ion_list=None):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_lnfugcoef_inf_dil(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, eps=eps, species=species, user_params=user_params, ion_list=ion_list)


def pcsaft_fugcoef_inf_dil(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p', eps=0.0, ion_list=None):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_fugcoef_inf_dil(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, eps=eps, species=species, user_params=user_params, ion_list=ion_list)


def pcsaft_gsolv(t, p, x, params=None, species=None, user_params=None, dielc_rule=None, ion_list=None):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_gsolv(t, p, x, params, dielc_rule=dielc_rule, species=species, user_params=user_params, ion_list=ion_list)


def pcsaft_gtransfer(t, p, x1, x2, params1=None, params2=None, species1=None, species2=None,
                     user_params1=None, user_params2=None, dielc_rule1=None, dielc_rule2=None, ion_list=None):
    params1 = _resolve_params(species1, t, user_params1, params1)
    params2 = _resolve_params(species2, t, user_params2, params2)
    return _pcsaft.pcsaft_gtransfer(t, p, x1, x2, params1, params2, dielc_rule1=dielc_rule1, dielc_rule2=dielc_rule2,
                                    species1=species1, species2=species2, user_params1=user_params1, user_params2=user_params2,
                                    ion_list=ion_list)


def pcsaft_actcoeff(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p', eps=0.0):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_actcoeff(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, eps=eps, species=species, user_params=user_params)


def pcsaft_miac(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p', eps=0.0):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_miac(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, eps=eps, species=species, user_params=user_params)


def pcsaft_miac_m(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p', eps=0.0):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_miac_m(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, eps=eps, species=species, user_params=user_params)


def pcsaft_debug_dielc(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_debug_dielc(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_debug_bjerrum(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_debug_bjerrum(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_salt_molality_to_ionic_x(t, x, params=None, species=None, user_params=None):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_salt_molality_to_ionic_x(t, x, params, species=species, user_params=user_params)


def pcsaft_hres(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_hres(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_sres(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_sres(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_gres(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_gres(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_ares(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p', return_contributions=False):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_ares(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params, return_contributions=return_contributions)


def pcsaft_ares_contrib(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_ares_contrib(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_dadt(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_dadt(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_osmoticC(t, p_or_rho, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_osmoticC(t, p_or_rho, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_cp(t, p_or_rho, aly_lee_params, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, input='p'):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_cp(t, p_or_rho, aly_lee_params, x, params, phase=phase, dielc_rule=dielc_rule, input=input, species=species, user_params=user_params)


def pcsaft_Hvap(t, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, p_guess=None):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.pcsaft_Hvap(t, x, params, p_guess=p_guess, phase=phase, dielc_rule=dielc_rule, species=species, user_params=user_params)


def flashTQ(t, q, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, p_guess=None):
    params = _resolve_params(species, t, user_params, params)
    return _pcsaft.flashTQ(t, q, x, params, p_guess=p_guess, phase=phase, dielc_rule=dielc_rule, species=species, user_params=user_params)


def flashPQ(p, q, x, params=None, species=None, user_params=None, phase='liq', dielc_rule=None, t_guess=None):
    params = _resolve_params(species, t_guess, user_params, params)
    return _pcsaft.flashPQ(p, q, x, params, t_guess=t_guess, phase=phase, dielc_rule=dielc_rule, species=species, user_params=user_params)


def aly_lee(t, c):
    return _pcsaft.aly_lee(t, c)


def dielc_water(t):
    return _pcsaft.dielc_water(t)


__all__ = [
    'get_prop_dict',
    'validate_species_params',
    'InputError', 'SolutionError',
    'pcsaft_den', 'pcsaft_p', 'pcsaft_Z',
    'pcsaft_Z_contrib',
    'pcsaft_ion_dh_debug',
    'pcsaft_lnfugcoef', 'pcsaft_fugcoef',
    'pcsaft_mu_res_contrib', 'pcsaft_mures',
    'pcsaft_lnfugcoef_inf_dil', 'pcsaft_fugcoef_inf_dil',
    'pcsaft_actcoeff', 'pcsaft_miac', 'pcsaft_miac_m', 'pcsaft_debug_dielc', 'pcsaft_debug_bjerrum',
    'pcsaft_salt_molality_to_ionic_x',
    'pcsaft_gsolv', 'pcsaft_gtransfer',
    'pcsaft_hres', 'pcsaft_sres', 'pcsaft_gres', 'pcsaft_ares', 'pcsaft_ares_contrib', 'pcsaft_dadt',
    'pcsaft_osmoticC', 'pcsaft_cp', 'pcsaft_Hvap',
    'flashTQ', 'flashPQ',
    'aly_lee', 'dielc_water',
]
