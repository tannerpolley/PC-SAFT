# -*- coding: utf-8 -*-
# setuptools: language=c++

import math
import numpy as np
from scipy.optimize import least_squares
from libcpp.vector cimport vector
from copy import deepcopy
from . cimport pcsaft

class InputError(Exception):
    # Exception raised for errors in the input.
    def __init__(self, message):
        self.message = message

class SolutionError(Exception):
    # Exception raised when a solver does not return a value.
    def __init__(self, message):
        self.message = message

def check_input(x, vars):
    if abs(np.sum(x) - 1) > 1e-7:
        raise InputError('The mole fractions do not sum to 1. x = {}'.format(x))
    if 'temperature' in vars:
        if vars['temperature'] <= 0:
            raise InputError('The {} must be a positive number. {} = {}'.format('temperature', 'temperature', vars['temperature']))
    if 'density' in vars:
        if vars['density'] <= 0:
            raise InputError('The {} must be a positive number. {} = {}'.format('density', 'density', vars['density']))
    if 'pressure' in vars:
        if vars['pressure'] <= 0:
            raise InputError('The {} must be a positive number. {} = {}'.format('pressure', 'pressure', vars['pressure']))
    if 'Q' in vars:
        if (vars['Q'] < 0) or (vars['Q'] > 1):
            raise InputError('{} must be <= 1 and >= 0. {} = {}'.format('Q', 'Q', vars['Q']))

def check_association(params):
    if ('e_assoc' in params) and ('vol_a' not in params):
        raise InputError('e_assoc was given, but not vol_a.')
    elif ('vol_a' in params) and ('e_assoc' not in params):
        raise InputError('vol_a was given, but not e_assoc.')

    if ('e_assoc' in params) and ('assoc_scheme' not in params):
        params['assoc_scheme'] = []
        for a in params['vol_a']:
            if a != 0:
                params['assoc_scheme'].append('2b')
            else:
                params['assoc_scheme'].append(None)

    if ('e_assoc' in params):
        params = create_assoc_matrix(params)

    return params

def create_assoc_matrix(params):
    charge = [] # whether the association site has a partial positive charge (i.e. hydrogen), negative charge, or elements of both (e.g. for acids modelled as type 1)

    scheme_charges = {
        '1': [0],
        '2a': [0, 0],
        '2b': [-1, 1],
        '3a': [0, 0, 0],
        '3b': [-1, -1, 1],
        '4a': [0, 0, 0, 0],
        '4b': [1, 1, 1, -1],
        '4c': [-1, -1, 1, 1]
    }

    assoc_num = []
    for comp in params['assoc_scheme']:
        if comp is None:
            assoc_num.append(0)
            pass
        elif type(comp) is list:
            num = 0
            for site in comp:
                if site.lower() not in scheme_charges:
                    raise InputError('{} is not a valid association type.'.format(site))
                charge.extend(scheme_charges[site.lower()])
                num += len(scheme_charges[site.lower()])
            assoc_num.append(num)
        else:
            if comp.lower() not in scheme_charges:
                raise InputError('{} is not a valid association type.'.format(comp))
            charge.extend(scheme_charges[comp.lower()])
            assoc_num.append(len(scheme_charges[comp.lower()]))
    params['assoc_num'] = np.asarray(assoc_num)

    params['assoc_matrix'] = np.zeros((len(charge)*len(charge)))
    ctr = 0
    for c1 in charge:
        for c2 in charge:
            if (c1 == 0 or c2 == 0):
                params['assoc_matrix'][ctr] = 1;
            elif (c1 == 1 and c2 == -1):
                params['assoc_matrix'][ctr] = 1;
            elif (c1 == -1 and c2 == 1):
                params['assoc_matrix'][ctr] = 1;
            else:
                params['assoc_matrix'][ctr] = 0;
            ctr += 1

    return params

def ensure_numpy_input(x, params):
    if np.isscalar(x):
        x = np.asarray([x], dtype=float)
    if np.isscalar(params['m']):
        params['m'] = np.asarray([params['m']], dtype=float)
    if np.isscalar(params['s']):
        params['s'] = np.asarray([params['s']], dtype=float)
    if np.isscalar(params['e']):
        params['e'] = np.asarray([params['e']], dtype=float)
    return x, params

def _safe_unit_log(values, floor=1e-300):
    vals = np.asarray(values, dtype=float)
    return np.log(np.maximum(vals, floor))


def _softmax(u):
    u = np.asarray(u, dtype=float)
    if u.size == 0:
        return u
    um = float(np.max(u))
    ex = np.exp(np.clip(u - um, -700.0, 700.0))
    s = float(np.sum(ex))
    if s <= 0.0 or (not np.isfinite(s)):
        return np.full(u.shape, 1.0/u.size, dtype=float)
    return ex/s


def _sigmoid(v):
    if v >= 0:
        e = math.exp(-v)
        return 1.0/(1.0 + e)
    e = math.exp(v)
    return e/(1.0 + e)


def _logit(v):
    vv = min(max(v, 1e-12), 1.0 - 1e-12)
    return math.log(vv/(1.0 - vv))


def _split_ion_groups(z, z_feed):
    z = np.asarray(z, dtype=float).flatten()
    z_feed = np.asarray(z_feed, dtype=float).flatten()
    if z.size != z_feed.size:
        raise InputError("z and z_feed must have the same length.")
    charged_idx = np.where(np.abs(z) > 1e-12)[0]
    if charged_idx.size == 0:
        raise InputError("pcsaft_multiphase_lle requires ionic species (non-zero z).")
    z_ch = z[charged_idx]
    cat_local = [int(i) for i in np.where(z_ch > 0.0)[0]]
    an_local = [int(i) for i in np.where(z_ch < 0.0)[0]]
    if len(cat_local) == 0 or len(an_local) == 0:
        raise InputError("pcsaft_multiphase_lle needs at least one cation and one anion.")
    cat_local = sorted(cat_local, key=lambda i: -float(z_feed[charged_idx[i]]))
    an_local = sorted(an_local, key=lambda i: -float(z_feed[charged_idx[i]]))
    return {
        "charged_idx": charged_idx.astype(int),
        "cat_local": cat_local,
        "an_local": an_local,
    }


def _build_e_matrix(z, z_feed, species=None):
    """
    Build Ascani 2022 E matrix for independent ionic pairs.
    Returns (E_matrix, info_dict).
    """
    split = _split_ion_groups(z, z_feed)
    charged_idx = split["charged_idx"]
    cat_local = split["cat_local"]
    an_local = split["an_local"]
    z = np.asarray(z, dtype=float).flatten()
    z_ch = z[charged_idx]
    n_ch = int(charged_idx.size)
    n_cat = len(cat_local)
    n_an = len(an_local)
    E = np.zeros((n_ch - 1, n_ch), dtype=float)

    if n_cat <= n_an:
        for k in range(n_an):
            E[k, cat_local[0]] = 1.0/abs(z_ch[cat_local[0]])
            E[k, an_local[k]] = 1.0/abs(z_ch[an_local[k]])
        if n_cat > 1:
            for k in range(n_cat - 1):
                row = n_an + k
                E[row, cat_local[k + 1]] = 1.0/abs(z_ch[cat_local[k + 1]])
                E[row, an_local[k]] = 1.0/abs(z_ch[an_local[k]])
    else:
        for k in range(n_cat):
            E[k, an_local[0]] = 1.0/abs(z_ch[an_local[0]])
            E[k, cat_local[k]] = 1.0/abs(z_ch[cat_local[k]])
        if n_an > 1:
            for k in range(n_an - 1):
                row = n_cat + k
                E[row, an_local[k + 1]] = 1.0/abs(z_ch[an_local[k + 1]])
                E[row, cat_local[k]] = 1.0/abs(z_ch[cat_local[k]])

    rank = int(np.linalg.matrix_rank(E))
    if rank != n_ch - 1:
        raise SolutionError("Failed to construct full-rank E matrix. rank={} expected={}.".format(rank, n_ch - 1))

    if species is None:
        species = [str(i) for i in range(z.size)]

    ion_pair_rows = []
    for r in range(E.shape[0]):
        cols = np.where(np.abs(E[r]) > 0.0)[0]
        ion_pair_rows.append({
            "row": int(r),
            "charged_local_indices": [int(c) for c in cols.tolist()],
            "species_indices": [int(charged_idx[c]) for c in cols.tolist()],
            "species": [species[int(charged_idx[c])] for c in cols.tolist()],
            "weights": [float(E[r, c]) for c in cols.tolist()],
        })

    info = {
        "charged_idx": charged_idx.astype(int),
        "charged_species": [species[int(i)] for i in charged_idx.tolist()],
        "rank": rank,
        "ion_pair_rows": ion_pair_rows,
    }
    return E, info


def _trial_x_from_n_xi(n_neut, xi, neutral_idx, charged_idx, E, ncomp):
    x = np.zeros(ncomp, dtype=float)
    n_neut = np.asarray(n_neut, dtype=float).flatten()
    xi = np.asarray(xi, dtype=float).flatten()
    if E.size == 0:
        n_ch = np.zeros(len(charged_idx), dtype=float)
    else:
        n_ch = E.T.dot(xi)
    n_ch = np.maximum(n_ch, 0.0)
    denom = float(np.sum(n_neut) + np.sum(n_ch))
    if denom <= 0.0:
        raise SolutionError("Trial composition denominator is non-positive.")
    if len(neutral_idx) > 0:
        x[np.asarray(neutral_idx, dtype=int)] = n_neut/denom
    if len(charged_idx) > 0:
        x[np.asarray(charged_idx, dtype=int)] = n_ch/denom
    x_sum = float(np.sum(x))
    if x_sum <= 0.0:
        raise SolutionError("Trial composition sum is non-positive.")
    return x/x_sum


def _phase_state_liq(t, p, x, params):
    rho = float(pcsaft_den(t, p, x, params, phase='liq'))
    lnfugcoef = np.asarray(pcsaft_lnfugcoef(t, rho, x, params), dtype=float)
    lnfug = lnfugcoef + _safe_unit_log(x) + math.log(float(p))
    return {"rho": rho, "lnfugcoef": lnfugcoef, "lnfug": lnfug}


def _tpdf_value(t, p, x_trial, lnfug_feed, params):
    state_trial = _phase_state_liq(t, p, x_trial, params)
    val = float(np.dot(x_trial, state_trial["lnfug"] - lnfug_feed))
    return val, state_trial


def _find_tpdf_seed(t, p, z_feed, params, neutral_idx, charged_idx, E, options):
    global_trials = int(options.get("tpdf_global_trials", 4000))
    local_trials = int(options.get("tpdf_local_trials", 2000))
    if global_trials <= 0:
        raise ValueError("tpdf_global_trials must be positive.")
    if local_trials < 0:
        raise ValueError("tpdf_local_trials must be >= 0.")

    rng = np.random.default_rng(int(options.get("random_seed", 12345)))
    ncomp = int(len(z_feed))
    feed_state = _phase_state_liq(t, p, z_feed, params)
    lnfug_feed = feed_state["lnfug"]

    n_neut_dim = len(neutral_idx)
    xi_dim = len(charged_idx) - 1
    best_val = np.inf
    best_state = None
    best_x = None
    best_n = None
    best_xi = None

    for _ in range(global_trials):
        n_neut = rng.random(n_neut_dim) + 1e-14
        xi = rng.random(xi_dim) + 1e-14
        try:
            x_trial = _trial_x_from_n_xi(n_neut, xi, neutral_idx, charged_idx, E, ncomp)
            tpdf, state = _tpdf_value(t, p, x_trial, lnfug_feed, params)
        except Exception:
            continue
        if np.isfinite(tpdf) and (tpdf < best_val):
            best_val = tpdf
            best_state = state
            best_x = x_trial
            best_n = n_neut.copy()
            best_xi = xi.copy()

    if best_x is None:
        raise SolutionError("TPDF search failed to find a valid trial phase.")

    if local_trials > 0:
        n_scale = np.maximum(best_n, 1e-3)
        xi_scale = np.maximum(best_xi, 1e-3)
        step_n = 0.30*n_scale
        step_xi = 0.30*xi_scale
        for itr in range(local_trials):
            cand_n = np.maximum(best_n + rng.normal(0.0, step_n, size=n_neut_dim), 1e-14)
            cand_xi = np.maximum(best_xi + rng.normal(0.0, step_xi, size=xi_dim), 1e-14)
            try:
                x_trial = _trial_x_from_n_xi(cand_n, cand_xi, neutral_idx, charged_idx, E, ncomp)
                tpdf, state = _tpdf_value(t, p, x_trial, lnfug_feed, params)
            except Exception:
                tpdf = np.inf
            if np.isfinite(tpdf) and (tpdf < best_val):
                best_val = tpdf
                best_state = state
                best_x = x_trial
                best_n = cand_n
                best_xi = cand_xi
            damp = 0.995 if itr < (local_trials//2) else 0.999
            step_n *= damp
            step_xi *= damp

    return {
        "tpdf_min": float(best_val),
        "seed_x": np.asarray(best_x, dtype=float),
        "seed_state": best_state,
        "feed_state": feed_state,
        "lnfug_feed": lnfug_feed,
    }


def _residual_two_phase(y, t, p, z_feed, params, z, E, neutral_idx, charged_idx, beta_bounds, charge_weight=1.0, anchor=None):
    ncomp = int(len(z_feed))
    nres = ncomp + (1 if anchor is not None else 0)
    beta_lo, beta_hi = float(beta_bounds[0]), float(beta_bounds[1])
    x1_logits = np.asarray(y[:ncomp - 1], dtype=float)
    x1 = _softmax(np.concatenate([x1_logits, np.zeros(1, dtype=float)]))
    beta = beta_lo + (beta_hi - beta_lo)*_sigmoid(float(y[ncomp - 1]))
    denom = (1.0 - beta)
    penalty = np.full(nres, 1e6, dtype=float)
    if denom <= 0.0:
        return penalty

    x2 = (z_feed - beta*x1)/denom
    if np.any(~np.isfinite(x2)):
        return penalty
    if np.any(x2 <= 1e-15):
        return penalty
    if abs(float(np.sum(x2)) - 1.0) > 1e-8:
        return penalty

    try:
        st1 = _phase_state_liq(t, p, x1, params)
        st2 = _phase_state_liq(t, p, x2, params)
    except Exception:
        return penalty

    res_neutral = st1["lnfug"][neutral_idx] - st2["lnfug"][neutral_idx]
    delta_ch = st1["lnfug"][charged_idx] - st2["lnfug"][charged_idx]
    res_ionic = E.dot(delta_ch)
    res_charge = np.array([float(charge_weight*np.dot(z, x1))], dtype=float)
    res = np.concatenate([res_neutral, res_ionic, res_charge])
    if anchor is not None:
        idx = int(anchor["component"])
        target = float(anchor["target"])
        weight = float(anchor.get("weight", 1.0))
        res = np.concatenate([res, np.array([weight*(x1[idx] - target)], dtype=float)])
    if res.size != nres:
        out = np.full(nres, 1e6, dtype=float)
        out[:min(nres, res.size)] = res[:min(nres, res.size)]
        return out
    return res


def _phase_equilibrium_residual(x1, x2, z, E, neutral_idx, charged_idx, lnfug1, lnfug2):
    dlnf = np.asarray(lnfug1, dtype=float) - np.asarray(lnfug2, dtype=float)
    res_neutral = dlnf[neutral_idx]
    res_ionic = E.dot(dlnf[charged_idx])
    res_charge = np.array([float(np.dot(z, x1))], dtype=float)
    return np.concatenate([res_neutral, res_ionic, res_charge])


def _solve_two_phase_lle(t, p, z_feed, params, z, E, neutral_idx, charged_idx, seed_x, options):
    solver_tol = float(options.get("solver_tol", 1e-9))
    max_nfev = int(options.get("max_nfev", 200))
    beta_bounds = options.get("beta_bounds", (1e-8, 1.0 - 1e-8))
    if len(beta_bounds) != 2:
        raise ValueError("beta_bounds must be a two-element tuple.")
    beta_lo, beta_hi = float(beta_bounds[0]), float(beta_bounds[1])
    charge_weight = float(options.get("charge_weight", 1000.0))
    charge_tol = float(options.get("charge_tol", 1e-6))
    mass_balance_tol = float(options.get("mass_balance_tol", 1e-8))
    split_tol = float(options.get("split_tol", 1e-3))
    solver_accept_norm = float(options.get("solver_accept_norm", 2e-1))
    if not (0.0 < beta_lo < beta_hi < 1.0):
        raise ValueError("beta_bounds must satisfy 0 < low < high < 1.")

    ncomp = int(len(z_feed))
    rng = np.random.default_rng(int(options.get("random_seed", 12345)))
    starts = []
    starts.append(np.asarray(seed_x, dtype=float))
    starts.append(np.asarray(z_feed, dtype=float))
    starts.append(0.8*np.asarray(seed_x, dtype=float) + 0.2*np.asarray(z_feed, dtype=float))
    starts.append(0.2*np.asarray(seed_x, dtype=float) + 0.8*np.asarray(z_feed, dtype=float))
    for _ in range(8):
        blend = float(rng.uniform(0.1, 0.9))
        jitter = rng.random(ncomp) + 1e-10
        starts.append(blend*np.asarray(seed_x, dtype=float) + (1.0 - blend)*jitter/np.sum(jitter))

    candidates = []
    for x_start in starts:
        x_start = np.maximum(x_start, 1e-12)
        x_start = x_start/np.sum(x_start)
        beta0 = 0.5*(beta_lo + beta_hi)
        y0 = np.zeros(ncomp, dtype=float)
        y0[:ncomp - 1] = np.log(np.maximum(x_start[:ncomp - 1], 1e-15)/max(float(x_start[-1]), 1e-15))
        frac = (beta0 - beta_lo)/(beta_hi - beta_lo)
        y0[ncomp - 1] = _logit(frac)
        try:
            sol = least_squares(
                _residual_two_phase,
                y0,
                args=(t, p, z_feed, params, z, E, neutral_idx, charged_idx, (beta_lo, beta_hi), charge_weight, None),
                method="trf",
                ftol=solver_tol,
                xtol=solver_tol,
                gtol=solver_tol,
                max_nfev=max_nfev,
            )
        except Exception:
            continue

        y = sol.x
        x1 = _softmax(np.concatenate([y[:ncomp - 1], np.zeros(1, dtype=float)]))
        beta = beta_lo + (beta_hi - beta_lo)*_sigmoid(float(y[ncomp - 1]))
        if beta <= 0.0 or beta >= 1.0:
            continue
        x2 = (z_feed - beta*x1)/(1.0 - beta)
        if np.any(~np.isfinite(x2)):
            continue
        if np.any(x2 <= 1e-15):
            continue
        if abs(float(np.sum(x2)) - 1.0) > 1e-8:
            continue
        try:
            st1 = _phase_state_liq(t, p, x1, params)
            st2 = _phase_state_liq(t, p, x2, params)
            residual = _phase_equilibrium_residual(x1, x2, z, E, neutral_idx, charged_idx, st1["lnfug"], st2["lnfug"])
        except Exception:
            continue

        candidates.append({
            "sol": sol,
            "x1": x1,
            "x2": x2,
            "beta": float(beta),
            "st1": st1,
            "st2": st2,
            "residual": residual,
            "residual_norm": float(np.linalg.norm(residual)),
            "split_norm": float(np.max(np.abs(x1 - x2))),
        })

    if len(candidates) == 0:
        return {
            "converged": False,
            "status": -1,
            "message": "least_squares did not produce a candidate solution.",
            "residual_norm": np.inf,
            "phases": [],
            "n_phases": 2,
        }

    best_res = min(c["residual_norm"] for c in candidates)
    split_candidates = [
        c for c in candidates
        if (c["split_norm"] > 1e-3 and c["residual_norm"] <= max(1e-3, 50.0*best_res))
    ]
    if len(split_candidates) > 0:
        best_cand = min(split_candidates, key=lambda c: (c["residual_norm"], c["sol"].cost))
    else:
        best_cand = min(candidates, key=lambda c: (c["residual_norm"], c["sol"].cost))

    # If unconstrained solve falls back to a trivial split, try a seeded anchored solve.
    if best_cand["split_norm"] <= 1e-4:
        split_delta = np.abs(np.asarray(seed_x, dtype=float) - np.asarray(z_feed, dtype=float))
        if split_delta.size > 0 and float(np.max(split_delta)) > 1e-8:
            anchor_idx = int(np.argmax(split_delta))
            anchor = {"component": anchor_idx, "target": float(seed_x[anchor_idx]), "weight": 1.0}
            anchored_candidates = []
            for x_start in starts:
                x_start = np.maximum(np.asarray(x_start, dtype=float), 1e-12)
                x_start = x_start/np.sum(x_start)
                beta0 = 0.5*(beta_lo + beta_hi)
                y0 = np.zeros(ncomp, dtype=float)
                y0[:ncomp - 1] = np.log(np.maximum(x_start[:ncomp - 1], 1e-15)/max(float(x_start[-1]), 1e-15))
                frac = (beta0 - beta_lo)/(beta_hi - beta_lo)
                y0[ncomp - 1] = _logit(frac)
                try:
                    sol = least_squares(
                        _residual_two_phase,
                        y0,
                        args=(t, p, z_feed, params, z, E, neutral_idx, charged_idx, (beta_lo, beta_hi), charge_weight, anchor),
                        method="trf",
                        ftol=solver_tol,
                        xtol=solver_tol,
                        gtol=solver_tol,
                        max_nfev=max_nfev,
                    )
                except Exception:
                    continue

                y = sol.x
                x1 = _softmax(np.concatenate([y[:ncomp - 1], np.zeros(1, dtype=float)]))
                beta = beta_lo + (beta_hi - beta_lo)*_sigmoid(float(y[ncomp - 1]))
                if beta <= 0.0 or beta >= 1.0:
                    continue
                x2 = (z_feed - beta*x1)/(1.0 - beta)
                if np.any(~np.isfinite(x2)):
                    continue
                if np.any(x2 <= 1e-15):
                    continue
                if abs(float(np.sum(x2)) - 1.0) > 1e-8:
                    continue
                try:
                    st1 = _phase_state_liq(t, p, x1, params)
                    st2 = _phase_state_liq(t, p, x2, params)
                    residual = _phase_equilibrium_residual(x1, x2, z, E, neutral_idx, charged_idx, st1["lnfug"], st2["lnfug"])
                except Exception:
                    continue
                anchored_candidates.append({
                    "sol": sol,
                    "x1": x1,
                    "x2": x2,
                    "beta": float(beta),
                    "st1": st1,
                    "st2": st2,
                    "residual": residual,
                    "residual_norm": float(np.linalg.norm(residual)),
                    "split_norm": float(np.max(np.abs(x1 - x2))),
                })

            anchored_candidates = [c for c in anchored_candidates if c["split_norm"] > 1e-4]
            if len(anchored_candidates) > 0:
                best_cand = min(anchored_candidates, key=lambda c: (c["residual_norm"], c["sol"].cost))
    best = best_cand["sol"]
    x1 = best_cand["x1"]
    x2 = best_cand["x2"]
    beta = best_cand["beta"]
    st1 = best_cand["st1"]
    st2 = best_cand["st2"]
    residual = best_cand["residual"]
    residual_norm = best_cand["residual_norm"]
    tol_ok = residual_norm <= max(solver_accept_norm, 1e3*solver_tol)
    charge_ok = abs(float(np.dot(z, x1))) <= charge_tol and abs(float(np.dot(z, x2))) <= charge_tol
    mb_ok = np.max(np.abs(z_feed - (beta*x1 + (1.0-beta)*x2))) <= mass_balance_tol
    frac_ok = (beta > beta_lo) and (beta < beta_hi)
    split_ok = best_cand["split_norm"] > split_tol
    converged = bool(charge_ok and mb_ok and frac_ok and split_ok and tol_ok)

    phases = [
        {
            "beta": float(beta),
            "x": np.asarray(x1, dtype=float),
            "rho": float(st1["rho"]),
            "lnfugcoef": np.asarray(st1["lnfugcoef"], dtype=float),
            "lnfug": np.asarray(st1["lnfug"], dtype=float),
        },
        {
            "beta": float(1.0 - beta),
            "x": np.asarray(x2, dtype=float),
            "rho": float(st2["rho"]),
            "lnfugcoef": np.asarray(st2["lnfugcoef"], dtype=float),
            "lnfug": np.asarray(st2["lnfug"], dtype=float),
        },
    ]

    return {
        "converged": converged,
        "status": int(best.status),
        "message": str(best.message),
        "residual_norm": residual_norm,
        "phases": phases,
        "n_phases": 2,
        "solver_result": best,
    }


def pcsaft_multiphase_lle(t, p, z_feed, params, species, options=None):
    """
    Two-liquid-phase electrolyte flash using Ascani 2022-style mean-ionic conditions.

    Parameters
    ----------
    t : float
        Temperature (K)
    p : float
        Pressure (Pa)
    z_feed : array_like
        Overall species mole fractions (sum to 1), aligned with `species`.
    params : dict
        Standard PC-SAFT/ePC-SAFT parameter dictionary.
    species : list[str]
        Species names aligned with `z_feed`.
    options : dict, optional
        Solver controls:
        - tpdf_global_trials (default 4000)
        - tpdf_local_trials (default 2000)
        - tpdf_tol (default -1e-8)
        - beta_bounds (default (1e-8, 1-1e-8))
        - solver_tol (default 1e-9)
        - solver_accept_norm (default 2e-1)
        - max_nfev (default 200)
        - charge_weight (default 1000.0)
        - charge_tol (default 1e-6)
        - mass_balance_tol (default 1e-8)
        - split_tol (default 1e-3)
        - debug (default False)
        - seed_x (optional external composition seed to override the internal TPDF seed)
        - force_seed_solve (default False; if True and seed_x is provided, attempt the
          two-phase solve even when the feed stability test reports a single liquid phase)

    Returns
    -------
    dict
        Structured result containing phase compositions, fractions, densities, fugacity
        diagnostics, TPDF information, and E-matrix metadata.
    """
    if options is None:
        options = {}
    z_feed = np.asarray(z_feed, dtype=float).flatten()
    if len(species) != z_feed.size:
        raise InputError("`species` length ({}) must match z_feed length ({}).".format(len(species), z_feed.size))
    check_input(z_feed, {"temperature": t, "pressure": p})

    x_dummy, params = ensure_numpy_input(z_feed, params)
    params = check_association(params)
    z = np.asarray(params.get("z", []), dtype=float).flatten()
    if z.size != z_feed.size:
        raise InputError("params['z'] must be present and aligned with z_feed for pcsaft_multiphase_lle.")
    if np.allclose(z, 0.0):
        raise InputError("pcsaft_multiphase_lle requires ionic species (non-zero z).")

    feed_charge = float(np.dot(z, z_feed))
    if abs(feed_charge) > 1e-8:
        raise InputError("Feed must be electroneutral. Sum(z_i*z_feed_i) = {}".format(feed_charge))

    neutral_idx = np.where(np.abs(z) <= 1e-12)[0].astype(int)
    split = _split_ion_groups(z, z_feed)
    charged_idx = split["charged_idx"].astype(int)
    if charged_idx.size < 2:
        raise InputError("At least two charged species are required.")
    E, e_info = _build_e_matrix(z, z_feed, species=species)

    tpdf = _find_tpdf_seed(t, p, z_feed, params, neutral_idx, charged_idx, E, options)
    tpdf_tol = float(options.get("tpdf_tol", -1e-8))
    debug = bool(options.get("debug", False))
    external_seed = options.get("seed_x", None)
    force_seed_solve = bool(options.get("force_seed_solve", False))
    seed_x = np.asarray(external_seed, dtype=float).flatten() if external_seed is not None else None
    if seed_x is not None:
        if seed_x.size != z_feed.size:
            raise InputError("options['seed_x'] length ({}) must match z_feed length ({}).".format(seed_x.size, z_feed.size))
        if np.any(seed_x <= 0.0):
            raise InputError("options['seed_x'] must contain only positive entries.")
        seed_x = seed_x/np.sum(seed_x)
    if debug:
        print("[DEBUG multiphase] tpdf_min =", tpdf["tpdf_min"], "tol =", tpdf_tol)

    if tpdf["tpdf_min"] >= tpdf_tol and not (force_seed_solve and seed_x is not None):
        st = tpdf["feed_state"]
        return {
            "n_phases": 1,
            "phases": [{
                "beta": 1.0,
                "x": np.asarray(z_feed, dtype=float),
                "rho": float(st["rho"]),
                "lnfugcoef": np.asarray(st["lnfugcoef"], dtype=float),
                "lnfug": np.asarray(st["lnfug"], dtype=float),
            }],
            "tpdf_min": float(tpdf["tpdf_min"]),
            "tpdf_seed_x": np.asarray(tpdf["seed_x"], dtype=float),
            "converged": True,
            "status": 1,
            "message": "Feed is stable (single liquid phase).",
            "residual_norm": 0.0,
            "e_matrix": np.asarray(E, dtype=float),
            "ion_pair_rows": e_info["ion_pair_rows"],
            "charged_species": e_info["charged_species"],
            "charged_species_indices": e_info["charged_idx"],
        }

    solve = _solve_two_phase_lle(
        t, p, z_feed, params, z, E, neutral_idx, charged_idx, seed_x if seed_x is not None else tpdf["seed_x"], options
    )

    result = {
        "n_phases": int(solve["n_phases"]),
        "phases": solve["phases"],
        "tpdf_min": float(tpdf["tpdf_min"]),
        "tpdf_seed_x": np.asarray(tpdf["seed_x"], dtype=float),
        "converged": bool(solve["converged"]),
        "status": int(solve["status"]),
        "message": solve["message"],
        "residual_norm": float(solve["residual_norm"]),
        "e_matrix": np.asarray(E, dtype=float),
        "ion_pair_rows": e_info["ion_pair_rows"],
        "charged_species": e_info["charged_species"],
        "charged_species_indices": e_info["charged_idx"],
    }
    return result


def pcsaft_p(t, rho, x, params):
    """
    Calculate pressure.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : ndarray, shape (n,)
            Component dielectric constants used for electrolyte calculations.
        MW : ndarray, shape (n,)
            Molecular weights in kg/mol; required for non-water solvent
            molality conversion.
        osmotic_solvent_index : int, optional
            Explicit solvent index for the molality basis. If omitted, the
            method chooses a neutral component (z=0) automatically.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    P : float
        Pressure (Pa)
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)
    return pcsaft_p_cpp(t, rho, x, cppargs)


def pcsaft_lnfugcoef(t, rho, x, params):
    """
    Calculate the natural logarithm of the fugacity coefficients for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    lnfugcoef : ndarray, shape (n,)
        Natural logarithm of the fugacity coefficients for each component.
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)
    return np.asarray(pcsaft_lnfug_cpp(t, rho, x, cppargs))


def pcsaft_lnfugcoef_terms(t, rho, x, params):
    """
    Calculate per-term residual chemical-potential contributions, per-term residual
    compressibility-factor pieces, intermediate derivative pieces, and
    contribution-resolved ln fugacity coefficients.

    Returns
    -------
    dict
        Keys map to arrays of shape (n,):
        - mu_hc, mu_disp, mu_polar, mu_assoc, mu_ion, mu_born
        - mu_total
        - lnfugcoef_hc, lnfugcoef_disp, lnfugcoef_polar, lnfugcoef_assoc
        - lnfugcoef_ion, lnfugcoef_born
        - lnfugcoef_total (alias: lnfugcoef)
        - dadx_hc, dadx_disp, dadx_polar, dadx_assoc, dadx_ion, dadx_born
        Scalar keys:
        - a_hc, a_disp, a_polar, a_assoc, a_ion, a_born
        - sum_x_dadx_hc, sum_x_dadx_disp, sum_x_dadx_polar, sum_x_dadx_assoc
        - sum_x_dadx_ion, sum_x_dadx_born
        - z_raw_hc, z_raw_disp, z_raw_polar, z_raw_assoc, z_raw_ion, z_raw_born
        - z_hc, z_disp, z_polar, z_assoc, z_ion, z_born
        - z_total

    Notes
    -----
    Contribution-resolved solvation and transfer energies should prefer the
    contribution-level ``lnfugcoef_*`` terms over the raw ``mu_*`` terms.
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)

    flat = np.asarray(pcsaft_lnfug_terms_cpp(t, rho, x, cppargs), dtype=float)
    ncomp = int(np.asarray(x, dtype=float).size)
    expected = 20 * ncomp + 25
    if flat.size != expected:
        raise SolutionError('Unexpected lnfug term payload size: expected {}, got {}.'.format(expected, int(flat.size)))

    blocks = flat[:20 * ncomp].reshape((20, ncomp))
    scalars = flat[20 * ncomp:]
    out = {
        'mu_hc': blocks[0],
        'mu_disp': blocks[1],
        'mu_polar': blocks[2],
        'mu_assoc': blocks[3],
        'mu_ion': blocks[4],
        'mu_born': blocks[5],
        'mu_total': blocks[6],
        'lnfugcoef_total': blocks[7],
        'lnfugcoef': blocks[7],
        'lnfugcoef_hc': blocks[8],
        'lnfugcoef_disp': blocks[9],
        'lnfugcoef_polar': blocks[10],
        'lnfugcoef_assoc': blocks[11],
        'lnfugcoef_ion': blocks[12],
        'lnfugcoef_born': blocks[13],
        'dadx_hc': blocks[14],
        'dadx_disp': blocks[15],
        'dadx_polar': blocks[16],
        'dadx_assoc': blocks[17],
        'dadx_ion': blocks[18],
        'dadx_born': blocks[19],
        'a_hc': float(scalars[0]),
        'a_disp': float(scalars[1]),
        'a_polar': float(scalars[2]),
        'a_assoc': float(scalars[3]),
        'a_ion': float(scalars[4]),
        'a_born': float(scalars[5]),
        'sum_x_dadx_hc': float(scalars[6]),
        'sum_x_dadx_disp': float(scalars[7]),
        'sum_x_dadx_polar': float(scalars[8]),
        'sum_x_dadx_assoc': float(scalars[9]),
        'sum_x_dadx_ion': float(scalars[10]),
        'sum_x_dadx_born': float(scalars[11]),
        'z_raw_hc': float(scalars[12]),
        'z_raw_disp': float(scalars[13]),
        'z_raw_polar': float(scalars[14]),
        'z_raw_assoc': float(scalars[15]),
        'z_raw_ion': float(scalars[16]),
        'z_raw_born': float(scalars[17]),
        'z_hc': float(scalars[18]),
        'z_disp': float(scalars[19]),
        'z_polar': float(scalars[20]),
        'z_assoc': float(scalars[21]),
        'z_ion': float(scalars[22]),
        'z_born': float(scalars[23]),
        'z_total': float(scalars[24]),
    }
    return out


def pcsaft_fugcoef(t, rho, x, params):
    """
    Calculate the fugacity coefficients for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    fugcoef : ndarray, shape (n,)
        Fugacity coefficients of each component.
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)
    return np.asarray(pcsaft_fugcoef_cpp(t, rho, x, cppargs))


def pcsaft_Z(t, rho, x, params):
    """
    Calculate the compressibility factor.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    Z : float
        Compressibility factor
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)
    return pcsaft_Z_cpp(t, rho, x, cppargs)


def flashPQ(p, q, x, params, t_guess=None):
    """
    Calculate the temperature of the system where vapor and liquid phases are in equilibrium.

    Parameters
    ----------
    p : float
        Pressure (Pa)
    q : float
        Mole fraction of the fluid in the vapor phase
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    t_guess : float
        Initial guess for the temperature (K) (optional)

    Returns
    -------
    t : float
        Temperature (K)
    xl : ndarray, shape (n,)
        Liquid mole fractions after flash
    xv : ndarray, shape (n,)
        Vapor mole fractions after flash

    Notes
    -----
    To solve the PQ flash the temperature must be varied. This adds additional complexity
    for water and electrolyte mixtures. For water, a temperature dependent sigma is often
    used. However, there does not appear to be a way to pass a Python function to the C++
    code without requiring the user to compile it using Cython. To avoid this, the `flashPQ`
    function uses the following relationship internally to calculate sigma for water as a
    function of temperature: ::

        3.8395 + 1.2828 * exp(-0.0074944 * t) - 1.3939 * exp(-0.00056029 * t);

    For electrolyte solutions the dielectric constant is calculated using the `dielc_water`
    function. This means that the sigma value for water and the dielectric constant given by
    the user are not used by the `flashPQ` function.

    The code identifies which component is water by the epsilon/k value. Therefore, when
    using `flashPQ` with water `e` must be exactly 353.9449, if you want the temperature
    dependence of sigma to be accounted for.

    If you want to use different functions for temperature dependent parameters with `flashPQ`
    then you will need to modify the source code and recompile it.
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'pressure':p, 'Q':q})
    params = check_association(params)
    cppargs = create_struct(params)
    try:
        if t_guess is not None:
            result = flashPQ_cpp(p, q, x, cppargs, t_guess)
        else:
            result = flashPQ_cpp(p, q, x, cppargs)
    except:
        raise SolutionError('A solution was not found for flashPQ. P={}'.format(p))

    t = result[0]
    xl = np.asarray(result[1:])
    xl, xv = np.split(xl, 2)
    return t, xl, xv


def flashTQ(t, q, x, params, p_guess=None):
    """
    Calculate the pressure of the system where vapor and liquid phases are in equilibrium.

    Parameters
    ----------
    t : float
        Temperature (K)
    q : float
        Mole fraction of the fluid in the vapor phase
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    p_guess : float
        Initial guess for the pressure (Pa) (optional)

    Returns
    -------
    p : float
        Pressure (Pa)
    xl : ndarray, shape (n,)
        Liquid mole fractions after flash
    xv : ndarray, shape (n,)
        Vapor mole fractions after flash
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'temperature':t, 'Q':q})
    params = check_association(params)
    cppargs = create_struct(params)
    try:
        if p_guess is not None:
            result = flashTQ_cpp(t, q, x, cppargs, p_guess)
        else:
            result = flashTQ_cpp(t, q, x, cppargs)
    except:
        raise SolutionError('A solution was not found for flashTQ. T={}'.format(t))

    p = result[0]
    xl = np.asarray(result[1:])
    xl, xv = np.split(xl, 2)
    return p, xl, xv

def pcsaft_Hvap(t, x, params, p_guess=None):
    """
    Calculate the enthalpy of vaporization.

    Parameters
    ----------
    t : float
        Temperature (K)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    p_guess : float
        Guess for the vapor pressure (Pa) (optional)

    Returns
    -------
    output : list
        A list containing the following results:
            0 : enthalpy of vaporization (J/mol), float
            1 : vapor pressure (Pa), float
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'temperature': t})
    params = check_association(params)
    cppargs = create_struct(params)

    q = 0
    try:
        if p_guess is not None:
            result = np.asarray(flashTQ_cpp(t, q, x, cppargs, p_guess))
            Pvap = result[0]
        else:
            result = np.asarray(flashTQ_cpp(t, q, x, cppargs))
            Pvap = result[0]
    except:
        raise SolutionError('A solution was not found for flashTQ. T={}'.format(t))

    rho = pcsaft_den_cpp(t, Pvap, x, 0, cppargs)
    hres_l = pcsaft_hres_cpp(t, rho, x, cppargs)
    rho = pcsaft_den_cpp(t, Pvap, x, 1, cppargs)
    hres_v = pcsaft_hres_cpp(t, rho, x, cppargs)
    Hvap = hres_v - hres_l

    output = [Hvap, Pvap]
    return output


def pcsaft_osmoticC(t, rho, x, params):
    """
    Calculate the osmotic coefficient.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    osmC : float
        Molal osmotic coefficient
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)

    # Select solvent component for molality basis.
    if 'osmotic_solvent_index' in params:
        indx_solvent = int(params['osmotic_solvent_index'])
        if indx_solvent < 0 or indx_solvent >= len(x):
            raise InputError('params["osmotic_solvent_index"] is out of bounds.')
    else:
        indx_solvent = -1
        z = np.asarray(params.get('z', []), dtype=float).flatten()
        if z.size == len(x):
            idx_sol = np.where(np.abs(z) < 1e-12)[0]
            if idx_sol.size == 1:
                indx_solvent = int(idx_sol[0])
            elif idx_sol.size > 1:
                # For mixed neutral solvents, default to the dominant neutral component.
                indx_solvent = int(idx_sol[np.argmax(x[idx_sol])])
        # Backward-compatible fallback for legacy water-only parameter sets.
        if indx_solvent < 0 and 'e' in params:
            idx_water = np.where(np.isclose(np.asarray(params['e'], dtype=float), 353.9449, atol=1e-6))[0]
            if idx_water.size == 1:
                indx_solvent = int(idx_water[0])
        if indx_solvent < 0:
            raise InputError('pcsaft_osmoticC requires a solvent component. Provide params["osmotic_solvent_index"] or one neutral species (z=0).')

    mw = np.asarray(params.get('MW', []), dtype=float).flatten()
    if mw.size == len(x):
        mw_solvent = float(mw[indx_solvent])
    elif 'e' in params and np.isclose(np.asarray(params['e'], dtype=float)[indx_solvent], 353.9449, atol=1e-6):
        # Legacy fallback for water if MW is omitted.
        mw_solvent = 18.0153/1000.
    else:
        raise InputError('pcsaft_osmoticC requires params["MW"] to compute molality for non-water solvent.')
    # If a molar mass in g/mol was passed by mistake, convert to kg/mol.
    if mw_solvent > 1.0:
        mw_solvent = mw_solvent/1000.0
    if mw_solvent <= 0.0:
        raise InputError('Solvent molecular weight must be positive.')
    if x[indx_solvent] <= 0.0:
        raise InputError('Solvent mole fraction must be positive.')

    molality = x/(x[indx_solvent]*mw_solvent)
    molality[indx_solvent] = 0
    molality_sum = float(np.sum(molality))
    if molality_sum <= 0.0:
        raise InputError('Total molality is zero; osmotic coefficient is undefined.')
    x0 = np.zeros_like(x)
    x0[indx_solvent] = 1.

    fugcoef = np.asarray(pcsaft_fugcoef_cpp(t, rho, x, cppargs))
    p = pcsaft_p_cpp(t, rho, x, cppargs)
    if rho < 900:
        ph = 1
    else:
        ph = 0
    rho0 = pcsaft_den_cpp(t, p, x0, ph, cppargs)
    fugcoef0 = np.asarray(pcsaft_fugcoef_cpp(t, rho0, x0, cppargs))
    gamma = float(fugcoef[indx_solvent]/fugcoef0[indx_solvent])

    osmC = -np.log(x[indx_solvent]*gamma)/(mw_solvent*molality_sum)
    # Keep legacy return shape for callers/tests that index result[0].
    return np.asarray([osmC], dtype=float)

def pcsaft_miac_m(t, rho, x, params, species=None):
    """
    Molality-scale mean ionic activity coefficient (MIAC_m).
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)

    z = np.asarray(params.get('z', []), dtype=float)
    if z.size == 0 or np.allclose(z, 0):
        raise InputError('pcsaft_miac_m requires ionic species (non-zero z).')
    idx_cat = np.where(z > 0)[0]
    idx_an = np.where(z < 0)[0]
    idx_sol = np.where(np.abs(z) < 1e-12)[0]
    if len(idx_cat) == 0 or len(idx_an) == 0:
        raise InputError('pcsaft_miac_m needs at least one cation and one anion.')
    if len(idx_sol) == 0:
        raise InputError('pcsaft_miac_m needs a neutral solvent to define molality.')

    mw = np.asarray(params['MW'], dtype=float)
    mass_solvent = float(np.sum(x[idx_sol] * mw[idx_sol]))
    if mass_solvent <= 0:
        raise InputError('Solvent mass is zero; check solvent mole fraction and MW.')

    fugcoef = np.asarray(pcsaft_fugcoef_cpp(t, rho, x, cppargs), dtype=float)

    eps = 1e-12
    x_inf = np.full_like(x, eps)
    solvent_ref = np.asarray(x[idx_sol], dtype=float)
    solvent_ref /= np.sum(solvent_ref)
    solvent_budget = max(1.0 - eps * (len(x) - len(idx_sol)), eps * len(idx_sol))
    x_inf[idx_sol] = solvent_ref * solvent_budget
    x_inf /= np.sum(x_inf)
    rho_inf = pcsaft_den_cpp(t, pcsaft_p_cpp(t, rho, x, cppargs), x_inf, 0, cppargs)
    fugcoef_inf = np.asarray(pcsaft_fugcoef_cpp(t, rho_inf, x_inf, cppargs), dtype=float)
    if np.any(fugcoef_inf <= 0):
        raise SolutionError('Non-positive fugacity at infinite dilution.')
    gamma_i = fugcoef / fugcoef_inf

    mass_neutral = x[idx_sol] * mw[idx_sol]
    w_sf = mass_neutral / mass_neutral.sum()
    M_solvent_mix = 1.0 / np.sum(w_sf / mw[idx_sol])

    if species is None or len(species) != len(x):
        raise InputError('species list (matching x order) is required to label salts.')

    result = {}
    for ic in idx_cat:
        for ia in idx_an:
            zc = int(round(abs(z[ic])))
            za = int(round(abs(z[ia])))
            g = math.gcd(zc, za)
            nu_cat = za // g
            nu_an = zc // g
            n_salt = 0.5 * (x[ic] / nu_cat + x[ia] / nu_an)
            m_salt = n_salt / mass_solvent
            sum_nu = float(nu_cat + nu_an)
            ln_gamma_pm = (nu_cat * math.log(gamma_i[ic]) + nu_an * math.log(gamma_i[ia])) / sum_nu
            gamma_pm_x = math.exp(ln_gamma_pm)
            gamma_pm_m = gamma_pm_x / (1.0 + M_solvent_mix * m_salt * sum_nu)
            salt_name = species[ic] + species[ia]
            result[salt_name] = gamma_pm_m

    return result

def pcsaft_miac(t, rho, x, params, species=None):
    """
    Mole-fraction-scale mean ionic activity coefficient (MIAC).
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)

    z = np.asarray(params.get('z', []), dtype=float)
    if z.size == 0 or np.allclose(z, 0):
        raise InputError('pcsaft_miac requires ionic species (non-zero z).')
    idx_cat = np.where(z > 0)[0]
    idx_an = np.where(z < 0)[0]
    idx_sol = np.where(np.abs(z) < 1e-12)[0]
    if len(idx_cat) == 0 or len(idx_an) == 0:
        raise InputError('pcsaft_miac needs at least one cation and one anion.')
    if len(idx_sol) == 0:
        raise InputError('pcsaft_miac needs a neutral solvent reference.')

    fugcoef = np.asarray(pcsaft_fugcoef_cpp(t, rho, x, cppargs), dtype=float)

    eps = 1e-12
    x_inf = np.full_like(x, eps)
    solvent_ref = np.asarray(x[idx_sol], dtype=float)
    solvent_ref /= np.sum(solvent_ref)
    solvent_budget = max(1.0 - eps * (len(x) - len(idx_sol)), eps * len(idx_sol))
    x_inf[idx_sol] = solvent_ref * solvent_budget
    x_inf /= np.sum(x_inf)
    rho_inf = pcsaft_den_cpp(t, pcsaft_p_cpp(t, rho, x, cppargs), x_inf, 0, cppargs)
    fugcoef_inf = np.asarray(pcsaft_fugcoef_cpp(t, rho_inf, x_inf, cppargs), dtype=float)
    if np.any(fugcoef_inf <= 0):
        raise SolutionError('Non-positive fugacity at infinite dilution.')
    gamma_i = fugcoef / fugcoef_inf

    if species is None or len(species) != len(x):
        raise InputError('species list (matching x order) is required to label salts.')

    result = {}
    for ic in idx_cat:
        for ia in idx_an:
            zc = int(round(abs(z[ic])))
            za = int(round(abs(z[ia])))
            g = math.gcd(zc, za)
            nu_cat = za // g
            nu_an = zc // g
            sum_nu = float(nu_cat + nu_an)
            ln_gamma_pm = (nu_cat * math.log(gamma_i[ic]) + nu_an * math.log(gamma_i[ia])) / sum_nu
            salt_name = species[ic] + species[ia]
            result[salt_name] = math.exp(ln_gamma_pm)

    return result

def pcsaft_cp(t, rho, aly_lee_params, x, params):
    """
    Calculate the specific molar isobaric heat capacity.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    aly_lee_params : ndarray, shape (5,)
        Constants for the Aly-Lee equation. Can be substituted with parameters for
        another equation if the ideal gas heat capacity is given using a different
        equation.
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each compopynent. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    cp : float
        Specific molar isobaric heat capacity (J mol\ :sup:`-1` K\ :sup:`-1`)
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)

    if rho > 900:
        ph = 0
    else:
        ph = 1

    cppargs = create_struct(params)

    cp_ideal = aly_lee(t, aly_lee_params)
    p = pcsaft_p_cpp(t, rho, x, cppargs)
    rho0 = pcsaft_den_cpp(t-0.001, p, x, ph, cppargs)
    hres0 = pcsaft_hres_cpp(t-0.001, rho0, x, cppargs)
    rho1 = pcsaft_den_cpp(t+0.001, p, x, ph, cppargs)
    hres1 = pcsaft_hres_cpp(t+0.001, rho1, x, cppargs)
    dhdt = (hres1-hres0)/0.002 # a numerical derivative is used for now until analytical derivatives are ready
    return cp_ideal + dhdt


def pcsaft_den(t, p, x, params, phase='liq'):
    """
    Calculate the molar density.

    Parameters
    ----------
    t : float
        Temperature (K)
    p : float
        Pressure (Pa)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    phase : string
        The phase for which the calculation is performed. Options: "liq" (liquid),
        "vap" (vapor).

    Returns
    -------
    rho : float
        Molar density (mol m\ :sup:`-3`)
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'pressure':p, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)
    if phase == 'liq':
        phase_num = 0
    else:
        phase_num = 1

    return pcsaft_den_cpp(t, p, x, phase_num, cppargs)


def pcsaft_hres(t, rho, x, params):
    """
    Calculate the residual enthalpy for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    hres : float
        Residual enthalpy (J mol\ :sup:`-1`)
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)
    return pcsaft_hres_cpp(t, rho, x, cppargs)

def pcsaft_sres(t, rho, x, params):
    """
    Calculate the residual entropy (constant volume) for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    sres : float
        Residual entropy (J mol\ :sup:`-1` K\ :sup:`-1`)
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)
    return pcsaft_sres_cpp(t, rho, x, cppargs)

def pcsaft_gres(t, rho, x, params):
    """
    Calculate the residual Gibbs energy for one phase of the system.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    gres : float
        Residual Gibbs energy (J mol\ :sup:`-1`)
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)
    return pcsaft_gres_cpp(t, rho, x, cppargs)

def pcsaft_gsolv(t, rho, x, params, species=None):
    """
    Gibbs solvation energy at infinite dilution on the mole-fraction scale for ions.
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)

    z = np.asarray(params.get('z', []), dtype=float)
    idx_ion = np.where(np.abs(z) > 1e-12)[0]
    if len(idx_ion) == 0:
        raise InputError('pcsaft_gsolv requires ionic species in params["z"].')
    if species is None or len(species) != len(x):
        raise InputError('species list (matching x order) is required to label ions.')

    idx_solv = np.where(np.abs(z) <= 1e-12)[0]
    if len(idx_solv) == 0:
        raise InputError('pcsaft_gsolv requires at least one solvent species (z=0).')

    x_ref = np.asarray(x, dtype=float).copy()
    x_ref[idx_ion] = 0.0
    solv_sum = np.sum(x_ref[idx_solv])
    if solv_sum > 0:
        x_ref[idx_solv] = x_ref[idx_solv] / solv_sum
    else:
        x_ref[idx_solv] = 1.0 / len(idx_solv)

    eps = 1e-12
    p = pcsaft_p_cpp(t, rho, x_ref, cppargs)
    phase = 1 if rho < 900 else 0
    result = {}
    for i in idx_ion:
        x_inf = x_ref.copy()
        x_inf[i] = eps
        x_inf /= np.sum(x_inf)
        rho_inf = pcsaft_den_cpp(t, p, x_inf, phase, cppargs)
        lnfug_inf = float(pcsaft_lnfug_cpp(t, rho_inf, x_inf, cppargs)[i])
        if not np.isfinite(lnfug_inf):
            raise SolutionError('Non-finite ln(fugacity coefficient) at infinite dilution.')
        result[species[i]] = 8.31446261815324 * t * lnfug_inf
    return result


def pcsaft_ares(t, rho, x, params):
    """
    Calculate the residual Helmholtz energy.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    ares : float
        Residual Helmholtz energy (J mol\ :sup:`-1`)
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)
    return pcsaft_ares_cpp(t, rho, x, cppargs)


def pcsaft_dadt(t, rho, x, params):
    """
    Calculate the temperature derivative of the residual Helmholtz energy.

    Parameters
    ----------
    t : float
        Temperature (K)
    rho : float
        Molar density (mol m\ :sup:`-3`)
    x : ndarray, shape (n,)
        Mole fractions of each component. It has a length of n, where n is
        the number of components in the system.
    params : dict
        A dictionary containing PC-SAFT parameters that can be passed for
        use in PC-SAFT:

        m : ndarray, shape (n,)
            Segment number for each component.
        s : ndarray, shape (n,)
            Segment diameter for each component. For ions this is the diameter of
            the hydrated ion. Units of Angstrom.
        e : ndarray, shape (n,)
            Dispersion energy of each component. For ions this is the dispersion
            energy of the hydrated ion. Units of K.
        k_ij : ndarray, shape (n,n)
            Binary interaction parameters between components in the mixture.
            (dimensions: ncomp x ncomp)
        e_assoc : ndarray, shape (n,)
            Association energy of the associating components. For non associating
            compounds this is set to 0. Units of K.
        vol_a : ndarray, shape (n,)
            Effective association volume of the associating components. For non
            associating compounds this is set to 0.
        dipm : ndarray, shape (n,)
            Dipole moment of the polar components. For components where the dipole
            term is not used this is set to 0. Units of Debye.
        dip_num : ndarray, shape (n,)
            The effective number of dipole functional groups on each component
            molecule. Generally this is set to 1, but some implementations use this
            as an adjustable parameter that is fit to data.
        z : ndarray, shape (n,)
            Charge number of the ions
        dielc : float
            Dielectric constant of the medium to be used for electrolyte
            calculations.
        assoc_scheme : list, shape (n,)
            The types of association sites for each component. Use `None` for molecules
            without association sites. If a molecule has multiple association sites,
            use a nested list for that component to specify the association scheme for
            each site. The accepted association schemes are those given by Huang and
            Radosz (1990): 1, 2A, 2B, 3A, 3B, 4A, 4B, 4C. If `e_assoc` and `vol_a` are
            given but `assoc_scheme` is not, the 2B association scheme is assumed (which
            would, for example, correspond to one hydroxyl functional group).

    Returns
    -------
    dadt : float
        Temperature derivative of the residual Helmholtz energy (J mol\ :sup:`-1`)
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {'density':rho, 'temperature':t})
    params = check_association(params)
    cppargs = create_struct(params)
    return pcsaft_dadt_cpp(t, rho, x, cppargs)


def aly_lee(t, c):
    """
    Calculate the ideal gas isobaric heat capacity using the Aly-Lee equation.

    Parameters
    ----------
    t : float
        Temperature (K)
    c : ndarray, shape (5,)
        Constants for the Aly-Lee equation

    Returns
    -------
    cp_ideal : float
        Ideal gas isobaric heat capacity (J mol\ :sup:`-1` K\ :sup:`-1`)

    References
    ----------
    - F. A. Aly and L. L. Lee, “Self-consistent equations for calculating the ideal gas heat capacity, enthalpy, and entropy,” Fluid Phase Equilibria, vol. 6, no. 3–4, pp. 169–179, 1981.
    """
    cp_ideal = (c[0] + c[1]*(c[2]/t/np.sinh(c[2]/t))**2 + c[3]*(c[4]/t/np.cosh(c[4]/t))**2)/1000.
    return cp_ideal

def dielc_water(t):
    """
    Return the dielectric constant of water at the given temperature.

    This equation was fit to values given in the reference. For temperatures from
    263.15 to 368.15 K values at 1 bar were used. For temperatures from 368.15 to
    443.15 K values at 10 bar were used. Below 263.15 K and above 443.15 K an
    error is raised.

    Parameters
    ----------
    t : float
        Temperature (K)

    Returns
    -------
    dielc : float
        Dielectric constant of water

    References
    ----------
    - D. G. Archer and P. Wang, “The Dielectric Constant of Water and Debye‐Hückel Limiting Law Slopes,” J. Phys. Chem. Ref. Data, vol. 19, no. 2, pp. 371–411, Mar. 1990.
    """
    if t < 263.15:
        raise ValueError('For dielc_water t must be greater than 263.15 K.')
    elif t > 443.15:
        raise ValueError('For dielc_water t must be less than 443.15 K.')

    if t <= 368.15:
        dielc = 7.6555618295E-04*t**2 - 8.1783881423E-01*t + 2.5419616803E+02
    else:
        dielc = 0.0005003272124*t**2 - 0.6285556029*t + 220.4467027
    return dielc


def np_to_vector_double(np_array):
    """Take a numpy array and return a C++ vector."""
    cdef vector[double] cpp_vector

    try:
        np_array = np_array.flatten()
        N = np_array.shape[0]
        for i in range(N):
            cpp_vector.push_back(np_array[i])
    except TypeError:
        cpp_vector.push_back(np_array)

    return cpp_vector

def np_to_vector_int(np_array):
    """Take a numpy array and return a C++ vector."""
    cdef vector[int] cpp_vector

    try:
        np_array = np_array.flatten()
        N = np_array.shape[0]
        for i in range(N):
            cpp_vector.push_back(np_array[i])
    except TypeError:
        cpp_vector.push_back(np_array)

    return cpp_vector

def create_struct(params):
    """Convert PC-SAFT parameters to a C++ struct."""
    cdef add_args cppargs
    cdef int ncomp

    cppargs.mixed_rel_perm_water_index = -1
    cppargs.m = np_to_vector_double(params['m'])
    ncomp = len(np.asarray(params['m']).flatten())
    cppargs.s = np_to_vector_double(params['s'])
    cppargs.e = np_to_vector_double(params['e'])
    if 'k_ij' in params:
        cppargs.k_ij = np_to_vector_double(params['k_ij'])
    if ('e_assoc' in params) and np.any(params['e_assoc']):
        cppargs.e_assoc = np_to_vector_double(params['e_assoc'])
    if ('vol_a' in params) and np.any(params['vol_a']):
        cppargs.vol_a = np_to_vector_double(params['vol_a'])
    if ('dipm' in params) and np.any(params['dip_num']) and np.any(params['dipm']):
        cppargs.dipm = np_to_vector_double(params['dipm'])
    if ('dip_num' in params) and np.any(params['dip_num']):
        cppargs.dip_num = np_to_vector_double(params['dip_num'])
    z_arr = None
    if 'z' in params:
        z_arr = np.asarray(params['z'], dtype=float).flatten()
        if z_arr.size not in (0, ncomp):
            raise ValueError('params["z"] must have length {} (or be empty), got {}.'.format(ncomp, z_arr.size))
        if z_arr.size == ncomp:
            cppargs.z = np_to_vector_double(z_arr)
    if 'dielc' in params:
        dielc_arr = np.asarray(params['dielc'], dtype=float).flatten()
        if dielc_arr.size != ncomp:
            raise ValueError('params["dielc"] must have length {}, got {}.'.format(ncomp, dielc_arr.size))
        cppargs.dielc = np_to_vector_double(dielc_arr)
    if 'MW' in params:
        mw_arr = np.asarray(params['MW'], dtype=float).flatten()
        if mw_arr.size != ncomp:
            raise ValueError('params["MW"] must have length {}, got {}.'.format(ncomp, mw_arr.size))
        cppargs.mw = np_to_vector_double(mw_arr)
    if 'mixed_rel_perm_a' in params:
        mixed_a_arr = np.asarray(params['mixed_rel_perm_a'], dtype=float).flatten()
        if mixed_a_arr.size != ncomp:
            raise ValueError('params["mixed_rel_perm_a"] must have length {}, got {}.'.format(ncomp, mixed_a_arr.size))
        cppargs.mixed_rel_perm_a = np_to_vector_double(mixed_a_arr)
    if 'mixed_rel_perm_b' in params:
        mixed_b_arr = np.asarray(params['mixed_rel_perm_b'], dtype=float).flatten()
        if mixed_b_arr.size != ncomp:
            raise ValueError('params["mixed_rel_perm_b"] must have length {}, got {}.'.format(ncomp, mixed_b_arr.size))
        cppargs.mixed_rel_perm_b = np_to_vector_double(mixed_b_arr)
    if 'mixed_rel_perm_c' in params:
        mixed_c_arr = np.asarray(params['mixed_rel_perm_c'], dtype=float).flatten()
        if mixed_c_arr.size != ncomp:
            raise ValueError('params["mixed_rel_perm_c"] must have length {}, got {}.'.format(ncomp, mixed_c_arr.size))
        cppargs.mixed_rel_perm_c = np_to_vector_double(mixed_c_arr)
    if 'mixed_rel_perm_mask' in params:
        mixed_mask_arr = np.asarray(params['mixed_rel_perm_mask'], dtype=int).flatten()
        if mixed_mask_arr.size != ncomp:
            raise ValueError('params["mixed_rel_perm_mask"] must have length {}, got {}.'.format(ncomp, mixed_mask_arr.size))
        cppargs.mixed_rel_perm_mask = np_to_vector_int(mixed_mask_arr)
    if 'mixed_rel_perm_water_index' in params:
        cppargs.mixed_rel_perm_water_index = int(params['mixed_rel_perm_water_index'])
    if cppargs.z.size() > 0 and cppargs.dielc.size() == 0:
        raise ValueError('Electrolyte parameters require params["dielc"] as a per-species array.')
    d_born_arr = None
    if 'd_born' in params:
        d_born_arr = np.asarray(params['d_born'], dtype=float).flatten()
        if d_born_arr.size != ncomp:
            raise ValueError('params["d_born"] must have length {}, got {}.'.format(ncomp, d_born_arr.size))
        cppargs.d_born = np_to_vector_double(d_born_arr)
    if 'f_solv' in params:
        cppargs.f_solv = np_to_vector_double(np.asarray(params['f_solv'], dtype=float))

    legacy_elec_keys = {
        'dielc_rule', 'dielc_diff_mode', 'born_model', 'born_radius_model',
        'born_diff_mode', 'born_eps_mode', 'DH_model', 'bjeruum_treatment',
        'd_ion_mode', 'include_born_model', 'd_Born_mode',
        'born_solvation_shell_model', 'born_dielectric_saturation', 'born_bulk_mode',
        'mu_DH_diff_mode', 'mu_DH_comp_dep_rel_perm', 'mu_DH_include_sum_term',
        'mu_born_diff_mode', 'mu_born_comp_dep_rel_perm', 'mu_born_include_sum_term', 'mu_born_comp_dep_delta_d'
    }
    if any((k in params) for k in legacy_elec_keys):
        raise ValueError(
            'Flat electrostatic params are no longer supported; provide nested params["elec_model"] schema.'
        )

    elec_model = params.get('elec_model', None)
    if elec_model is None:
        if cppargs.z.size() > 0:
            # Backward-compatible electrolyte defaults when no explicit user options are provided.
            elec_model = {
                'rel_perm': {'rule': 1, 'differential_mode': 'analytical'},
                'hc_model': {'dadx_differential_mode': 'analytical'},
                'disp_model': {'dadx_differential_mode': 'analytical'},
                'assoc_model': {'dadx_differential_mode': 'analytical'},
                'polar_model': {'dadx_differential_mode': 'analytical'},
                'DH_model': {
                    'd_ion_mode': 1,
                    'bjeruum_treatment': False,
                    'mu_DH_model': {
                        'differential_mode': 'analytical',
                        'comp_dep_rel_perm': True,
                        'include_sum_term': True,
                    },
                },
                'include_born_model': True,
                'born_model': {
                    'd_Born_mode': 0,
                    'solvation_shell_model': False,
                    'dielectric_saturation': False,
                    'bulk_mode': 'mix',
                    'mu_born_model': {
                        'differential_mode': 'analytical',
                        'comp_dep_rel_perm': True,
                        'include_sum_term': True,
                        'comp_dep_delta_d': False,
                    },
                },
            }
        else:
            elec_model = {}
    if not isinstance(elec_model, dict):
        raise ValueError('params["elec_model"] must be a dict when provided.')

    def _as_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, np.integer)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {'1', 'true', 'yes', 'y', 'on'}:
                return True
            if s in {'0', 'false', 'no', 'n', 'off'}:
                return False
        raise ValueError('Could not coerce value to bool: {}'.format(v))

    def _as_int_alias(v, aliases):
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in aliases:
                return int(aliases[s])
            if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
                return int(s)
        raise ValueError('Unknown option value: {}'.format(v))

    rule_alias = {
        'constant': 0,
        'rule0': 0,
        'linear': 1,
        'linear-molefraction': 1,
        'linear-mixing-mole': 1,
        'rule1': 1,
        'linear-massfraction': 2,
        'linear-mixing-weight': 2,
        'rule2': 2,
        'combined': 3,
        'rule3': 3,
        'empirical': 4,
        'rule4': 4,
        'rule5': 5,
        'rule6': 6,
        'aqueous-organic': 8,
        'aqueous_organic': 8,
        'mixed-aqueous-organic': 8,
        'mixed_aqueous_organic': 8,
        'rule8': 8,
    }
    diff_alias = {'analytic': 0, 'analytical': 0, 'numeric': 1, 'numerical': 1, 'autodiff': 2}
    try:
        cppargs.dadt_diff_mode = _as_int_alias(params.get('dadt_differential_mode', 'analytical'), diff_alias)
    except ValueError:
        raise ValueError('Unknown dadt_differential_mode. Supported values are analytical/numerical/autodiff (0/1/2).')
    if cppargs.dadt_diff_mode not in (0, 1, 2):
        raise ValueError('Unknown dadt_differential_mode. Supported values are analytical/numerical/autodiff (0/1/2).')
    d_ion_alias = {'t_indep': 0, 't_dep_1': 1, 't_dep_2': 2}
    d_born_alias = {'t_indep': 0, 't_dep_1': 1, 't_dep_2': 2, 'fitted_param': 3}
    bulk_alias = {'mix': 0, 'bulk': 0, 'solvent': 1}

    rel_perm = elec_model.get('rel_perm', {})
    if not isinstance(rel_perm, dict):
        raise ValueError('params["elec_model"]["rel_perm"] must be a dict.')
    hc_model_dict = elec_model.get('hc_model', {})
    if not isinstance(hc_model_dict, dict):
        raise ValueError('params["elec_model"]["hc_model"] must be a dict.')
    disp_model_dict = elec_model.get('disp_model', {})
    if not isinstance(disp_model_dict, dict):
        raise ValueError('params["elec_model"]["disp_model"] must be a dict.')
    assoc_model_dict = elec_model.get('assoc_model', {})
    if not isinstance(assoc_model_dict, dict):
        raise ValueError('params["elec_model"]["assoc_model"] must be a dict.')
    polar_model_dict = elec_model.get('polar_model', {})
    if not isinstance(polar_model_dict, dict):
        raise ValueError('params["elec_model"]["polar_model"] must be a dict.')
    dh_model_dict = elec_model.get('DH_model', {})
    if not isinstance(dh_model_dict, dict):
        raise ValueError('params["elec_model"]["DH_model"] must be a dict.')
    born_model_dict = elec_model.get('born_model', {})
    if not isinstance(born_model_dict, dict):
        raise ValueError('params["elec_model"]["born_model"] must be a dict.')
    mu_dh = dh_model_dict.get('mu_DH_model', {})
    if not isinstance(mu_dh, dict):
        raise ValueError('params["elec_model"]["DH_model"]["mu_DH_model"] must be a dict.')
    mu_born = born_model_dict.get('mu_born_model', {})
    if not isinstance(mu_born, dict):
        raise ValueError('params["elec_model"]["born_model"]["mu_born_model"] must be a dict.')

    cppargs.dielc_rule = _as_int_alias(rel_perm.get('rule', 1), rule_alias)
    cppargs.dielc_diff_mode = _as_int_alias(rel_perm.get('differential_mode', 'analytical'), diff_alias)
    if cppargs.dielc_diff_mode not in (0, 1, 2):
        raise ValueError('Unknown rel_perm differential_mode. Supported values are analytical/numerical/autodiff (0/1/2).')
    cppargs.hc_dadx_diff_mode = _as_int_alias(hc_model_dict.get('dadx_differential_mode', 'analytical'), diff_alias)
    if cppargs.hc_dadx_diff_mode not in (0, 1, 2):
        raise ValueError('Unknown hc_model dadx_differential_mode. Supported values are analytical/numerical/autodiff (0/1/2).')
    cppargs.disp_dadx_diff_mode = _as_int_alias(disp_model_dict.get('dadx_differential_mode', 'analytical'), diff_alias)
    if cppargs.disp_dadx_diff_mode not in (0, 1, 2):
        raise ValueError('Unknown disp_model dadx_differential_mode. Supported values are analytical/numerical/autodiff (0/1/2).')
    cppargs.assoc_dadx_diff_mode = _as_int_alias(assoc_model_dict.get('dadx_differential_mode', 'analytical'), diff_alias)
    if cppargs.assoc_dadx_diff_mode not in (0, 1, 2):
        raise ValueError('Unknown assoc_model dadx_differential_mode. Supported values are analytical/numerical/autodiff (0/1/2).')
    cppargs.polar_dadx_diff_mode = _as_int_alias(polar_model_dict.get('dadx_differential_mode', 'analytical'), diff_alias)
    if cppargs.polar_dadx_diff_mode not in (0, 1, 2):
        raise ValueError('Unknown polar_model dadx_differential_mode. Supported values are analytical/numerical/autodiff (0/1/2).')
    if cppargs.dielc_rule < 0 or cppargs.dielc_rule > 8:
        raise ValueError('Unknown rel_perm rule. Supported values are 0..8.')

    cppargs.d_ion_mode = _as_int_alias(dh_model_dict.get('d_ion_mode', 1), d_ion_alias)
    if cppargs.d_ion_mode not in (0, 1, 2):
        raise ValueError('Unknown d_ion_mode. Supported values are 0,1,2.')
    bjeruum = _as_bool(dh_model_dict.get('bjeruum_treatment', False))
    cppargs.mu_DH_diff_mode = _as_int_alias(mu_dh.get('differential_mode', 'analytical'), diff_alias)
    if cppargs.mu_DH_diff_mode not in (0, 1, 2):
        raise ValueError('Unknown mu_DH differential_mode. Supported values are analytical/numerical/autodiff (0/1/2).')
    cppargs.mu_DH_comp_dep_rel_perm = int(_as_bool(mu_dh.get('comp_dep_rel_perm', True)))
    cppargs.mu_DH_include_sum_term = int(_as_bool(mu_dh.get('include_sum_term', True)))

    cppargs.include_born_model = int(_as_bool(elec_model.get('include_born_model', True)))
    cppargs.d_born_mode = _as_int_alias(born_model_dict.get('d_Born_mode', 0), d_born_alias)
    if cppargs.d_born_mode not in (0, 1, 2, 3):
        raise ValueError('Unknown d_Born_mode. Supported values are 0,1,2,3.')
    cppargs.born_solvation_shell_model = int(_as_bool(born_model_dict.get('solvation_shell_model', False)))
    cppargs.born_dielectric_saturation = int(_as_bool(born_model_dict.get('dielectric_saturation', False)))
    cppargs.born_bulk_mode = _as_int_alias(born_model_dict.get('bulk_mode', 'mix'), bulk_alias)
    cppargs.mu_born_diff_mode = _as_int_alias(mu_born.get('differential_mode', 'analytical'), diff_alias)
    if cppargs.mu_born_diff_mode not in (0, 1, 2):
        raise ValueError('Unknown mu_born differential_mode. Supported values are analytical/numerical/autodiff (0/1/2).')
    cppargs.mu_born_comp_dep_rel_perm = int(_as_bool(mu_born.get('comp_dep_rel_perm', True)))
    cppargs.mu_born_include_sum_term = int(_as_bool(mu_born.get('include_sum_term', True)))
    cppargs.mu_born_comp_dep_delta_d = int(_as_bool(mu_born.get('comp_dep_delta_d', False)))

    if cppargs.include_born_model == 0:
        cppargs.born_model = 0
        cppargs.born_radius_model = 1
        cppargs.born_diff_mode = 0
        cppargs.born_eps_mode = cppargs.born_bulk_mode
    else:
        if cppargs.born_solvation_shell_model or cppargs.born_dielectric_saturation:
            cppargs.born_model = 2
        else:
            cppargs.born_model = 1

        # Legacy radius projection kept for C++ internals.
        if cppargs.d_born_mode == 0:
            cppargs.born_radius_model = 1
        elif cppargs.d_born_mode == 1:
            cppargs.born_radius_model = 2
        elif cppargs.d_born_mode == 2:
            cppargs.born_radius_model = 3
        else:
            cppargs.born_radius_model = 5

        cppargs.born_eps_mode = cppargs.born_bulk_mode

        if cppargs.mu_born_diff_mode == 1:
            cppargs.born_diff_mode = 1
        elif cppargs.mu_born_comp_dep_rel_perm == 0:
            cppargs.born_diff_mode = 3
        elif cppargs.mu_born_include_sum_term == 0:
            cppargs.born_diff_mode = 2
        else:
            cppargs.born_diff_mode = 0

    if cppargs.born_model == 1 and cppargs.born_radius_model == 5:
        raise ValueError('d_Born_mode="fitted_param" requires SSM/DS Born path (include_born_model=true and SSM or DS true).')

    if cppargs.born_model > 0 and cppargs.born_radius_model in (4, 5):
        if z_arr is None:
            raise ValueError("fitted d_Born_mode requires params['z'] as a per-species array.")
        if d_born_arr is None:
            raise ValueError("fitted d_Born_mode requires params['d_born'] as a per-species array.")
        ion_mask = np.abs(z_arr) > 1e-12
        if np.any(d_born_arr[ion_mask] <= 0.0):
            raise ValueError("fitted d_Born_mode requires positive ionic params['d_born'] values.")

    cppargs.DH_model = 2 if bjeruum else 1
    if cppargs.DH_model == 2:
        raise ValueError("Bjerrum treatment is reserved and not implemented (DH_model=2).")
    if cppargs.DH_model < 0 or cppargs.DH_model > 2:
        raise ValueError("Unknown DH_model. Supported values are 0, 1, and reserved 2.")
    cppargs.debug = int(bool(params['debug'])) if 'debug' in params else 0
    if 'assoc_num' in params:
        cppargs.assoc_num = np_to_vector_int(params['assoc_num'])
    if 'assoc_matrix' in params:
        cppargs.assoc_matrix = np_to_vector_int(params['assoc_matrix'])
    if 'k_hb' in params:
        cppargs.k_hb = np_to_vector_double(params['k_hb'])
    if 'l_ij' in params:
        cppargs.l_ij = np_to_vector_double(params['l_ij'])

    return cppargs


def _pcsaft_autodiff_residual_derivatives(t, rho, x, params):
    """Private development helper exposing autodiff residual derivatives."""
    x, params = ensure_numpy_input(x, params)
    params = check_association(params)
    cppargs = create_struct(params)
    ncomp = len(np.asarray(x).flatten())
    return {
        'dadt': pcsaft_dadt_cpp(t, rho, x, cppargs),
        'dadx': np.asarray(pcsaft_autodiff_dadx_cpp(t, rho, x, cppargs), dtype=float),
        'd2adt2': pcsaft_autodiff_d2adt2_cpp(t, rho, x, cppargs),
        'd2adtdx': np.asarray(pcsaft_autodiff_d2adtdx_cpp(t, rho, x, cppargs), dtype=float),
        'hessian_x': np.asarray(pcsaft_autodiff_hessian_x_cpp(t, rho, x, cppargs), dtype=float).reshape((ncomp, ncomp)),
    }


def pcsaft_dielc_eval(x, params):
    """
    Evaluate mixed dielectric constant and composition derivatives using the C++ dielectric engine.
    """
    x, params = ensure_numpy_input(x, params)
    check_input(x, {})
    params = check_association(params)
    cppargs = create_struct(params)
    eps = pcsaft_dielc_eps_cpp(x, cppargs)
    deps = np.asarray(pcsaft_dielc_diff_cpp(x, cppargs))
    return eps, deps

