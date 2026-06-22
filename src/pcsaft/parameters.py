"""Dataset-driven parameter loader for PC-SAFT/ePC-SAFT.

This module reads package-owned parameter sets from
``pcsaft.data/pcsaft_parameters/<dataset>/`` and returns dictionaries in the
same shape expected by ``pcsaft`` runtime calls.

Public API:
    - get_prop_dict(dataset_name, species, x, T, user_options=None)
    - available_datasets()
    - default_user_options()
    - minimize_user_options(user_options)
    - _resolve_runtime_options(user_options)
    - molality_to_molefraction(...)
    - molefraction_to_molality(...)
"""

from __future__ import annotations

import atexit
import copy
import csv
import json
import math
import re
from contextlib import ExitStack
from importlib.resources import as_file
from importlib.resources import files
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


_RESOURCE_STACK = ExitStack()
atexit.register(_RESOURCE_STACK.close)
DATASET_ROOT = _RESOURCE_STACK.enter_context(
    as_file(files("pcsaft.data").joinpath("pcsaft_parameters"))
)
DATASET_ROOT = Path(DATASET_ROOT)

BASE_KEYS = ["MW", "m", "s", "e", "e_assoc", "vol_a", "assoc_scheme", "dipm", "dip_num", "z", "dielc"]
OPTIONAL_KEYS = ["d_born", "f_solv"]

COMPONENT_ALIASES = {
    "H2O-2B-Li": "H2O",
    "H2O-2B-NaCl": "H2O",
    "H2O-Salt-2001": "H2O",
    "Water": "H2O",
    "water": "H2O",
    "methanol": "Methanol",
    "ethanol": "Ethanol",
    "1-Butanol": "Butanol",
    "butanol": "Butanol",
    "Isobutanol": "Butanol",
    "isobutanol": "Butanol",
    "benzene": "Benzene",
    "toluene": "Toluene",
    "glycine": "Glycine",
    "alanine": "Alanine",
    "propanol": "Propanol",
}

PURE_SET_KEY_ALIASES = {
    "water": "water",
    "h2o": "water",
    "methanol": "methanol",
    "ethanol": "ethanol",
    "any": "any_solvent",
    "default": "any_solvent",
    "any_solvent": "any_solvent",
}

_COMPONENT_DEFAULTS = {
    "H+": {"MW": 1.008e-3, "z": 1.0, "d_born": 1.0},
    "Li+": {"MW": 6.94e-3, "z": 1.0, "d_born": 2.784},
    "Na+": {"MW": 22.98e-3, "z": 1.0, "d_born": 3.445},
    "K+": {"MW": 39.10e-3, "z": 1.0, "d_born": 4.150},
    "NH4+": {"MW": 18.038e-3, "m": 1.0, "s": 3.5740, "e": 230.0, "z": 1.0, "d_born": 3.0},
    "F-": {"MW": 18.998e-3, "m": 1.0, "s": 1.7712, "e": 275.0, "z": -1.0, "d_born": 3.52},
    "Cl-": {"MW": 35.45e-3, "z": -1.0, "d_born": 4.100},
    "Br-": {"MW": 79.90e-3, "z": -1.0, "d_born": 4.480},
    "I-": {"MW": 126.90e-3, "z": -1.0, "d_born": 4.985},
    "Ac-": {"MW": 59.044e-3, "m": 1.0, "s": 4.3000, "e": 300.0, "z": -1.0, "d_born": 4.30},
    "H2O": {
        "MW": 18.01528e-3,
        "m": 1.2047,
        "s": lambda T: 2.7927 + 10.11 * np.exp(-0.01775 * T) - 1.417 * np.exp(-0.01146 * T),
        "e": 353.95,
        "e_assoc": 2425.7,
        "vol_a": 0.04509,
        "assoc_scheme": "2B",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 78.09,
        "d_born": 0.0,
        "f_solv": 1.5,
    },
    "Methanol": {
        "MW": 32.04e-3,
        "m": 1.5255,
        "s": 3.2300,
        "e": 188.90,
        "e_assoc": 2899.5,
        "vol_a": 0.03518,
        "assoc_scheme": "2B",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 33.05,
        "d_born": 0.0,
        "f_solv": 1.4,
    },
    "Ethanol": {
        "MW": 46.068e-3,
        "m": 2.3827,
        "s": 3.1771,
        "e": 198.24,
        "e_assoc": 2653.4,
        "vol_a": 0.03238,
        "assoc_scheme": "2B",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 24.88,
        "d_born": 0.0,
        "f_solv": 1.6,
    },
    "Propanol": {
        "MW": 60.095e-3,
        "m": 3.0,
        "s": 3.2522,
        "e": 233.40,
        "e_assoc": 2276.78,
        "vol_a": 0.01527,
        "assoc_scheme": "2B",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 20.47,
        "d_born": 0.0,
        "f_solv": 1.0,
    },
    "Butanol": {
        "MW": 74.1216e-3,
        "m": 2.7515,
        "s": 3.6139,
        "e": 259.59,
        "e_assoc": 2544.56,
        "vol_a": 0.00669,
        "assoc_scheme": "2B",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 17.51,
        "d_born": 0.0,
        "f_solv": 1.0,
    },
    "Benzene": {
        "MW": 78.1118e-3,
        "m": 2.4653,
        "s": 3.6478,
        "e": 287.35,
        "e_assoc": 0.0,
        "vol_a": 0.0,
        "assoc_scheme": "",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 2.28,
        "d_born": 0.0,
        "f_solv": 1.0,
    },
    "Toluene": {
        "MW": 92.1405e-3,
        "m": 2.8149,
        "s": 3.7169,
        "e": 285.69,
        "e_assoc": 0.0,
        "vol_a": 0.0,
        "assoc_scheme": "",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 2.38,
        "d_born": 0.0,
        "f_solv": 1.0,
    },
    "Glycine": {
        "MW": 75.067e-3,
        "m": 4.8507,
        "s": 2.3270,
        "e": 216.96,
        "e_assoc": 2598.1,
        "vol_a": 0.0393,
        "assoc_scheme": "2B",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 8.0,
        "d_born": 0.0,
        "f_solv": 1.0,
    },
    "Alanine": {
        "MW": 89.094e-3,
        "m": 5.4647,
        "s": 2.5222,
        "e": 287.59,
        "e_assoc": 3176.6,
        "vol_a": 0.0819,
        "assoc_scheme": "2B",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 8.0,
        "d_born": 0.0,
        "f_solv": 1.0,
    },
    "Hexane": {
        "MW": 86.178e-3,
        "m": 3.0578,
        "s": 3.7983,
        "e": 236.77,
        "e_assoc": 0.0,
        "vol_a": 0.0,
        "assoc_scheme": "",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 2.0,
        "d_born": 0.0,
        "f_solv": 1.0,
    },
    "Dodecane": {
        "MW": 170.334e-3,
        "m": 5.3060,
        "s": 3.8959,
        "e": 249.21,
        "e_assoc": 0.0,
        "vol_a": 0.0,
        "assoc_scheme": "",
        "dipm": 0.0,
        "dip_num": 1.0,
        "z": 0.0,
        "dielc": 2.0,
        "d_born": 0.0,
        "f_solv": 1.0,
    },
}

_LINEAR_T_RE = re.compile(
    r"^\s*([+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?(?:\*10\^[+\-]?\d+)?)\s*\*?\s*T(?:\s*/\s*K)?\s*([+\-]\s*\d*\.?\d+(?:[eE][+\-]?\d+)?)\s*$",
    flags=re.IGNORECASE,
)
_LOG_T_RE = re.compile(
    r"^\s*([+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?)\s*\*?\s*ln\s*\(\s*T\s*\)\s*([+\-]\s*\d*\.?\d+(?:[eE][+\-]?\d+)?)\s*$",
    flags=re.IGNORECASE,
)
_FLOAT_RE = re.compile(r"[+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?")

_REL_PERM_RULE_ALIASES = {
    "constant": 0,
    "rule0": 0,
    "linear": 1,
    "linear-molefraction": 1,
    "linear-mixing-mole": 1,
    "rule1": 1,
    "rule1a": 7,
    "linear-saltfraction": 7,
    "linear-mixing-salt": 7,
    "linear-massfraction": 2,
    "linear-mixing-weight": 2,
    "rule2": 2,
    "combined": 3,
    "rule3": 3,
    "empirical": 4,
    "rule4": 4,
    "rule5": 5,
    "rule6": 6,
    "aqueous-organic": 8,
    "aqueous_organic": 8,
    "mixed-aqueous-organic": 8,
    "mixed_aqueous_organic": 8,
    "rule8": 8,
}

SOLVENT_COMPONENT_TO_TOKEN = {
    "H2O": "water",
    "Methanol": "methanol",
    "Ethanol": "ethanol",
    "Propanol": "propanol",
    "Butanol": "butanol",
}

SOLVENT_TOKEN_TO_COMPONENT = {token: comp for comp, token in SOLVENT_COMPONENT_TO_TOKEN.items()}
SOLVENT_TOKEN_ORDER = {"water": 0, "methanol": 1, "ethanol": 2, "propanol": 3, "butanol": 4}
_DIFF_MODE_ALIASES = {
    "analytic": 0,
    "analytical": 0,
    "numeric": 1,
    "numerical": 1,
    "autodiff": 2,
}
_D_ION_MODE_ALIASES = {
    "t_indep": 0,
    "t_dep_1": 1,
    "t_dep_2": 2,
}
_D_BORN_MODE_ALIASES = {
    "t_indep": 0,
    "t_dep_1": 1,
    "t_dep_2": 2,
    "fitted_param": 3,
}
_BULK_MODE_ALIASES = {
    "mix": 0,
    "bulk": 0,
    "solvent": 1,
}
_CANONICAL_CONTRIBUTION_MODEL = {
    "dadx_differential_mode": "analytical",
}
_CANONICAL_ELEC_MODEL = {
    "rel_perm": {
        "rule": 1,
        "differential_mode": "analytical",
    },
    "hc_model": dict(_CANONICAL_CONTRIBUTION_MODEL),
    "disp_model": dict(_CANONICAL_CONTRIBUTION_MODEL),
    "assoc_model": dict(_CANONICAL_CONTRIBUTION_MODEL),
    "polar_model": dict(_CANONICAL_CONTRIBUTION_MODEL),
    "DH_model": {
        # Preserve current behavior (ionic diameter uses 0.88*sigma by default).
        "d_ion_mode": 1,
        "bjeruum_treatment": False,
        "mu_DH_model": {
            "differential_mode": "analytical",
            "comp_dep_rel_perm": True,
            "include_sum_term": True,
        },
    },
    "include_born_model": True,
    "born_model": {
        "d_Born_mode": 0,
        "solvation_shell_model": False,
        "dielectric_saturation": False,
        "bulk_mode": "mix",
        "mu_born_model": {
            "differential_mode": "analytical",
            "comp_dep_rel_perm": True,
            "include_sum_term": True,
            "comp_dep_delta_d": False,
        },
    },
}
_DEFAULT_USER_OPTIONS = {
    "debug": False,
    "dadt_differential_mode": "analytical",
    "solvated_ion_diameter_mixing_rule": False,
    "ion_dispersion_mixing_rule": True,
    "elec_model": copy.deepcopy(_CANONICAL_ELEC_MODEL),
}

_DATASET_CACHE: Dict[str, dict] = {}
_MISSING = object()


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot coerce value '{value}' to bool.")


def _deep_update(base: dict, updates: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _prune_default_overrides(payload, defaults):
    if isinstance(payload, dict):
        if not isinstance(defaults, dict):
            return copy.deepcopy(payload)
        pruned = {}
        for key, value in payload.items():
            default_value = defaults.get(key, _MISSING)
            if default_value is _MISSING:
                pruned[key] = copy.deepcopy(value)
                continue
            candidate = _prune_default_overrides(value, default_value)
            if candidate is not _MISSING:
                pruned[key] = candidate
        return pruned if pruned else _MISSING
    return _MISSING if payload == defaults else copy.deepcopy(payload)


def default_user_options() -> dict:
    """Return a deep copy of the package's canonical user-option defaults."""

    return copy.deepcopy(_DEFAULT_USER_OPTIONS)


def minimize_user_options(user_options: dict | None) -> dict:
    """Drop any user-option entries that are identical to package defaults."""

    if user_options is None:
        return {}
    if not isinstance(user_options, dict):
        raise TypeError("user_options must be a dict.")
    _resolve_runtime_options(user_options)
    pruned = _prune_default_overrides(user_options, _DEFAULT_USER_OPTIONS)
    return {} if pruned is _MISSING else pruned


def _normalize_component(name: str) -> str:
    return COMPONENT_ALIASES.get(name, name)


def available_datasets() -> list[str]:
    if not DATASET_ROOT.exists():
        return []
    return sorted(p.name for p in DATASET_ROOT.iterdir() if p.is_dir())


def _dataset_dir(dataset_name: str) -> Path:
    path = DATASET_ROOT / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Unknown dataset '{dataset_name}'. Available datasets: {available_datasets()}")
    return path


def _normalize_pure_set_key(name: str) -> str:
    return PURE_SET_KEY_ALIASES.get(name.strip().lower(), name.strip().lower())


def _solvent_token_for_component(name: str) -> str | None:
    component = _normalize_component(name.strip())
    return SOLVENT_COMPONENT_TO_TOKEN.get(component)


def _canonical_solvent_tokens(tokens: Iterable[str]) -> list[str]:
    unique = {str(token).strip().lower() for token in tokens if str(token).strip()}
    return sorted(unique, key=lambda token: (SOLVENT_TOKEN_ORDER.get(token, 999), token))


def _solvent_system_data_key(tokens: Iterable[str]) -> str:
    canonical = _canonical_solvent_tokens(tokens)
    return "-".join(canonical)


def _solvent_fraction_aliases(token: str, basis: str) -> tuple[str, ...]:
    token = str(token).strip().lower()
    if token == "water":
        names = ("water", "h2o")
    elif token == "methanol":
        names = ("methanol", "meoh")
    elif token == "ethanol":
        names = ("ethanol", "etoh")
    else:
        names = (token,)
    aliases: list[str] = []
    for name in names:
        aliases.append(f"{basis}_{name}")
        aliases.append(f"{basis}_{name}_salt_free")
    return tuple(aliases)


def _mixture_molecular_weight_from_token_fractions(x_map: dict[str, float]) -> float | None:
    total = 0.0
    for token, frac in x_map.items():
        component = SOLVENT_TOKEN_TO_COMPONENT.get(token)
        if component is None:
            return None
        mw = _deterministic_default(component, "MW", 298.15)
        if mw is _MISSING:
            return None
        total += float(frac) * float(mw)
    return total if total > 0.0 else None


def _convert_weight_to_mole_fractions(weights: dict[str, float]) -> dict[str, float] | None:
    numerators: dict[str, float] = {}
    for token, weight in weights.items():
        component = SOLVENT_TOKEN_TO_COMPONENT.get(token)
        if component is None:
            return None
        mw = _deterministic_default(component, "MW", 298.15)
        if mw is _MISSING or float(mw) <= 0.0:
            return None
        numerators[token] = float(weight) / float(mw)
    total = float(sum(numerators.values()))
    if total <= 0.0:
        return None
    return {token: value / total for token, value in numerators.items()}


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_component_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        comp = _normalize_component(str(row.get("component", "")).strip())
        if not comp:
            continue
        mapping[comp] = {k: str(v or "").strip() for k, v in row.items()}
    return mapping


def _load_pure_sets(pure_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    pure_sets: dict[str, dict[str, dict[str, str]]] = {}
    if not pure_dir.exists():
        return pure_sets
    for pure_file in sorted(pure_dir.glob("*.csv")):
        set_key = _normalize_pure_set_key(pure_file.stem)
        set_map = _load_component_rows(pure_file)
        if set_map:
            pure_sets[set_key] = set_map
    return pure_sets


def _select_default_pure_set_key(pure_sets: dict[str, dict[str, dict[str, str]]]) -> str | None:
    if "any_solvent" in pure_sets:
        return "any_solvent"
    if len(pure_sets) == 1:
        return next(iter(pure_sets))
    if "water" in pure_sets:
        return "water"
    return None


def _load_matrix(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    rows = _read_csv(path)
    if not rows:
        return {}
    columns = [c for c in rows[0].keys() if c and c != "component"]
    matrix = {}
    for row in rows:
        row_comp = _normalize_component(str(row.get("component", "")).strip())
        if not row_comp:
            continue
        for col in columns:
            col_comp = _normalize_component(col.strip())
            matrix[(row_comp, col_comp)] = str(row.get(col, "") or "").strip()
    return matrix


def _load_mixed_rel_perm(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows = _read_csv(path)
    mixed: dict[str, dict[str, float]] = {}
    for row in rows:
        organic = _normalize_component(str(row.get("organic", row.get("component", ""))).strip())
        if not organic:
            continue
        params: dict[str, float] = {}
        for key in ("a", "b", "c"):
            value = _maybe_float(row.get(key))
            if value is None:
                raise ValueError(f"Missing mixed rel_perm coefficient '{key}' for organic '{organic}'.")
            params[key] = float(value)
        mixed[organic] = params
    return mixed


def _load_specific_mixed_dielc_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    solvent_tokens = [token for token in path.stem.split("-") if token]
    rows = _read_csv(path)
    entries: list[dict] = []
    for row in rows:
        row_norm = {str(key).strip().lower(): value for key, value in row.items() if key}
        dielc = None
        for key in ("dielc", "rel_perm", "epsilon", "eps_r"):
            dielc = _maybe_float(row_norm.get(key))
            if dielc is not None:
                break
        if dielc is None:
            continue

        x_map: dict[str, float] = {}
        w_map: dict[str, float] = {}
        for token in solvent_tokens:
            x_val = None
            for alias in _solvent_fraction_aliases(token, "x"):
                x_val = _maybe_float(row_norm.get(alias.lower()))
                if x_val is not None:
                    break
            if x_val is not None:
                x_map[token] = float(x_val)

            w_val = None
            for alias in _solvent_fraction_aliases(token, "w"):
                w_val = _maybe_float(row_norm.get(alias.lower()))
                if w_val is not None:
                    break
            if w_val is not None:
                w_map[token] = float(w_val)

        if len(x_map) != len(solvent_tokens):
            if len(w_map) != len(solvent_tokens):
                continue
            converted = _convert_weight_to_mole_fractions(w_map)
            if converted is None:
                continue
            x_map = converted

        x_total = float(sum(x_map.values()))
        if x_total <= 0.0:
            continue
        x_norm = {token: value / x_total for token, value in x_map.items()}

        if len(w_map) != len(solvent_tokens):
            mw_mix = _mixture_molecular_weight_from_token_fractions(x_norm)
            if mw_mix is None:
                continue
            w_map = {}
            for token, frac in x_norm.items():
                component = SOLVENT_TOKEN_TO_COMPONENT[token]
                mw = float(_deterministic_default(component, "MW", 298.15))
                w_map[token] = frac * mw / mw_mix

        w_total = float(sum(w_map.values()))
        if w_total <= 0.0:
            continue
        w_norm = {token: value / w_total for token, value in w_map.items()}

        entries.append(
            {
                "x": x_norm,
                "w": w_norm,
                "dielc": float(dielc),
            }
        )
    return entries


def _load_mixed_rel_perm_tables(rel_perm_dir: Path) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {}
    if not rel_perm_dir.exists():
        return tables
    for table_file in sorted(rel_perm_dir.glob("*.csv")):
        if table_file.stem.lower() == "parameters":
            continue
        table = _load_specific_mixed_dielc_table(table_file)
        if table:
            tables[table_file.stem] = table
    return tables


def _strip_preset_keys(canonical: dict) -> dict:
    cleaned = copy.deepcopy(canonical)
    elec = cleaned.get("elec_model")
    if isinstance(elec, dict):
        elec.pop("preset", None)
        elec.pop("base", None)
    return cleaned


def _load_canonical_user_options(dataset_dir: Path) -> dict:
    path = dataset_dir / "user_options.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict) and "canonical_user_options" in payload:
        canonical = payload.get("canonical_user_options", {})
    else:
        canonical = payload

    if not isinstance(canonical, dict):
        return {}
    return _strip_preset_keys(canonical)


def _load_dataset(dataset_name: str) -> dict:
    if dataset_name in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_name]

    dataset_dir = _dataset_dir(dataset_name)

    pure_dir = dataset_dir / "pure"
    pure_sets = _load_pure_sets(pure_dir)

    pure_default_key = _select_default_pure_set_key(pure_sets)
    pure_map: dict[str, dict[str, str]] = {}
    if pure_default_key is not None:
        pure_map = pure_sets[pure_default_key]

    if not pure_map and not pure_sets:
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' must include pure/*.csv component-parameter files."
        )

    mixed_dir = dataset_dir / "mixed"
    bi_dir = mixed_dir / "binary_interaction"
    rel_perm_dir = mixed_dir / "rel_perm"
    rel_perm_coeff_path = rel_perm_dir / "parameters.csv"
    data = {
        "dataset_name": dataset_name,
        "dataset_dir": dataset_dir,
        "pure": pure_map,
        "pure_sets": pure_sets,
        "pure_default_key": pure_default_key,
        "k_ij": _load_matrix(bi_dir / "k_ij.csv"),
        "l_ij": _load_matrix(bi_dir / "l_ij.csv"),
        "k_hb": _load_matrix(bi_dir / "k_hb_ij.csv") or _load_matrix(bi_dir / "khb_ij.csv"),
        "mixed_rel_perm": _load_mixed_rel_perm(rel_perm_coeff_path),
        "mixed_rel_perm_tables": _load_mixed_rel_perm_tables(rel_perm_dir),
        "canonical_user_options": _load_canonical_user_options(dataset_dir),
    }
    _DATASET_CACHE[dataset_name] = data
    return data


def _maybe_float(raw) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (float, int, np.floating, np.integer)):
        value = float(raw)
        return value if math.isfinite(value) else None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        value = float(text)
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def _parse_t_coefficient(token: str) -> float:
    text = token.replace(" ", "")
    if "*10^" in text:
        mantissa, exponent = text.split("*10^", 1)
        return float(mantissa) * (10 ** int(exponent))
    return float(text)


def _parse_linear_t_expression(raw: str, T: float) -> float | None:
    text = raw.replace(" ", "")
    text = text.replace("/K", "").replace("/k", "")
    match = _LINEAR_T_RE.match(text)
    if not match:
        return None
    slope = _parse_t_coefficient(match.group(1))
    intercept = float(match.group(2).replace(" ", ""))
    return slope * T + intercept


def _parse_log_t_expression(raw: str, T: float) -> float | None:
    text = raw.replace(" ", "")
    match = _LOG_T_RE.match(text)
    if not match:
        return None
    coeff = float(match.group(1))
    intercept = float(match.group(2).replace(" ", ""))
    return coeff * np.log(T) + intercept


def _parse_water_sigma_expression(raw: str, T: float) -> float | None:
    text = raw.replace(" ", "").lower()
    if "2.7927" in text and "10.11" in text and "1.417" in text:
        return 2.7927 + 10.11 * np.exp(-0.01775 * T) - 1.417 * np.exp(-0.01146 * T)
    return None


def _parse_association_scheme(raw: str) -> str | None:
    text = str(raw or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _deterministic_default(component: str, prop: str, T: float):
    entry = _COMPONENT_DEFAULTS.get(component)
    if entry is not None and prop in entry:
        value = entry[prop]
        return value(T) if callable(value) else value

    is_ion = component.endswith("+") or component.endswith("-")
    if is_ion:
        if prop == "z":
            return 1.0 if component.endswith("+") else -1.0
        if prop == "m":
            return 1.0
        if prop in {"e_assoc", "vol_a", "dipm", "d_born"}:
            return 0.0
        if prop == "dip_num":
            return 1.0
        if prop == "assoc_scheme":
            return None
        if prop == "dielc":
            return 8.0
        if prop == "f_solv":
            return 1.0

    if prop == "assoc_scheme":
        return None

    # Generic charge inference for unknown ions.
    if prop == "z":
        if component.endswith("+"):
            return 1.0
        if component.endswith("-"):
            return -1.0
    return _MISSING


def _parse_cell_value(raw, *, dataset: str, component: str, field: str, T: float):
    if field == "assoc_scheme":
        return _parse_association_scheme(raw)

    numeric = _maybe_float(raw)
    if numeric is not None:
        return numeric

    text = str(raw or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    if field == "s" and component == "H2O":
        parsed = _parse_water_sigma_expression(text, T)
        if parsed is not None:
            return parsed

    log_t = _parse_log_t_expression(text, T)
    if log_t is not None:
        return log_t

    linear = _parse_linear_t_expression(text, T)
    if linear is not None:
        return linear

    if "=" in text:
        rhs = text.split("=")[-1].strip()
        rhs_numeric = _maybe_float(rhs)
        if rhs_numeric is not None:
            return rhs_numeric

    numbers = _FLOAT_RE.findall(text)
    if len(numbers) == 1:
        return float(numbers[0])

    raise ValueError(
        f"Unsupported value in dataset '{dataset}', component '{component}', field '{field}': '{text}'."
    )


def _resolve_component_field(dataset: dict, component: str, field: str, T: float, pure_set_key: str | None = None):
    row = None
    if pure_set_key:
        row = dataset.get("pure_sets", {}).get(_normalize_pure_set_key(pure_set_key), {}).get(component)
    if row is None:
        component_pure_key = _solvent_token_for_component(component)
        if component_pure_key is not None:
            row = dataset.get("pure_sets", {}).get(component_pure_key, {}).get(component)
    if row is None:
        default_key = dataset.get("pure_default_key")
        if default_key:
            row = dataset.get("pure_sets", {}).get(default_key, {}).get(component)
    if row is None:
        row = dataset["pure"].get(component)
    if row is None:
        fallback = _deterministic_default(component, field, T)
        if fallback is not _MISSING:
            return fallback
        raise KeyError(
            f"Component '{component}' is missing in dataset '{dataset['dataset_name']}' pure parameter files."
        )

    parsed = _parse_cell_value(
        row.get(field, ""),
        dataset=dataset["dataset_name"],
        component=component,
        field=field,
        T=T,
    )
    if parsed is not None:
        return parsed

    fallback = _deterministic_default(component, field, T)
    if fallback is not _MISSING:
        return fallback

    raise KeyError(
        f"Missing required value in dataset '{dataset['dataset_name']}', component '{component}', field '{field}'."
    )


def _resolve_component_field_with_source(
    dataset: dict,
    component: str,
    field: str,
    T: float,
    pure_set_key: str | None = None,
):
    requested_key = _normalize_pure_set_key(pure_set_key) if pure_set_key else None
    pure_sets = dataset.get("pure_sets", {})

    source_key = None
    row = None
    if requested_key:
        row = pure_sets.get(requested_key, {}).get(component)
        if row is not None:
            source_key = requested_key
    if row is None:
        component_pure_key = _solvent_token_for_component(component)
        if component_pure_key is not None:
            row = pure_sets.get(component_pure_key, {}).get(component)
            if row is not None:
                source_key = component_pure_key
    if row is None:
        default_key = dataset.get("pure_default_key")
        if default_key:
            row = pure_sets.get(default_key, {}).get(component)
            if row is not None:
                source_key = default_key
    if row is None:
        row = dataset["pure"].get(component)
        if row is not None and dataset.get("pure_default_key"):
            source_key = dataset["pure_default_key"]

    if row is None:
        fallback = _deterministic_default(component, field, T)
        if fallback is not _MISSING:
            return fallback, source_key
        raise KeyError(
            f"Component '{component}' is missing in dataset '{dataset['dataset_name']}' pure parameter files."
        )

    value = _resolve_component_field(dataset, component, field, T, pure_set_key=pure_set_key)
    return value, source_key


def _matrix_value(dataset: dict, matrix_name: str, c1: str, c2: str, T: float) -> float:
    matrix = dataset[matrix_name]
    raw = matrix.get((c1, c2))
    if raw is None:
        raw = matrix.get((c2, c1))
    if raw is None or not str(raw).strip():
        return 0.0
    value = _parse_cell_value(
        raw,
        dataset=dataset["dataset_name"],
        component=f"{c1}|{c2}",
        field=matrix_name,
        T=T,
    )
    if value is None:
        return 0.0
    if isinstance(value, str):
        raise ValueError(
            f"Non-numeric matrix value in dataset '{dataset['dataset_name']}' for pair '{c1}|{c2}' in {matrix_name}."
        )
    return float(value)


def _as_rule_number(value, aliases: dict[str, int]) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        key = value.strip().lower()
        if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
            return int(key)
        if key in aliases:
            return int(aliases[key])
    if not aliases and isinstance(value, str):
        key = value.strip().lower()
        if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
            return int(key)
    raise ValueError(f"Unknown rule option '{value}'. Supported aliases: {sorted(aliases.keys())}.")


def _resolve_born_bulk_mode(value) -> int:
    if isinstance(value, (int, np.integer)):
        mode = int(value)
        if mode in (0, 1):
            return mode
    key = str(value).strip().lower()
    if key in _BULK_MODE_ALIASES:
        return int(_BULK_MODE_ALIASES[key])
    raise ValueError("born bulk_mode must be one of {'mix','solvent'} (or 0/1).")


def _resolve_d_ion_mode(value) -> int:
    if isinstance(value, (int, np.integer)):
        mode = int(value)
    else:
        key = str(value).strip().lower()
        if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
            mode = int(key)
        elif key in _D_ION_MODE_ALIASES:
            mode = int(_D_ION_MODE_ALIASES[key])
        else:
            raise ValueError(
                f"Unknown d_ion_mode '{value}'. Supported values are 0,1,2 and aliases {sorted(_D_ION_MODE_ALIASES.keys())}."
            )
    if mode < 0 or mode > 2:
        raise ValueError("d_ion_mode must be in {0,1,2}.")
    return mode


def _resolve_d_born_mode(value) -> int:
    if isinstance(value, (int, np.integer)):
        mode = int(value)
    else:
        key = str(value).strip().lower()
        if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
            mode = int(key)
        elif key in _D_BORN_MODE_ALIASES:
            mode = int(_D_BORN_MODE_ALIASES[key])
        else:
            raise ValueError(
                f"Unknown d_Born_mode '{value}'. Supported values are 0,1,2,3 and aliases {sorted(_D_BORN_MODE_ALIASES.keys())}."
            )
    if mode < 0 or mode > 3:
        raise ValueError("d_Born_mode must be in {0,1,2,3}.")
    return mode


def _normalize_elec_model(model) -> dict:
    out = copy.deepcopy(_CANONICAL_ELEC_MODEL)
    if model is None:
        return out
    if not isinstance(model, dict):
        raise TypeError(f"elec_model must be a dict, got {type(model).__name__}.")

    # New nested schema blocks.
    if "rel_perm" in model:
        if not isinstance(model["rel_perm"], dict):
            raise TypeError("elec_model['rel_perm'] must be a dict.")
        out["rel_perm"] = _deep_update(out["rel_perm"], model["rel_perm"])

    for key in ("hc_model", "disp_model", "assoc_model", "polar_model"):
        if key in model:
            if not isinstance(model[key], dict):
                raise TypeError(f"elec_model['{key}'] must be a dict.")
            out[key] = _deep_update(out[key], model[key])

    if "DH_model" in model:
        if isinstance(model["DH_model"], dict):
            out["DH_model"] = _deep_update(out["DH_model"], model["DH_model"])
        elif isinstance(model["DH_model"], (int, np.integer, str)):
            # Legacy integer-style selector: 1/2 from older implementation.
            out["DH_model"]["bjeruum_treatment"] = int(model["DH_model"]) == 2
        else:
            raise TypeError("elec_model['DH_model'] must be a dict or legacy int/string.")

    if "include_born_model" in model:
        out["include_born_model"] = _coerce_bool(model["include_born_model"])

    if "born_model" in model:
        born_value = model["born_model"]
        if isinstance(born_value, dict):
            out["born_model"] = _deep_update(out["born_model"], born_value)
        elif isinstance(born_value, (int, np.integer, str)):
            # Legacy born_model int mapping.
            born_int = int(born_value)
            if born_int not in (0, 1, 2):
                raise ValueError("Legacy born_model must be one of 0, 1, 2.")
            out["include_born_model"] = born_int != 0
            out["born_model"]["solvation_shell_model"] = born_int == 2
            out["born_model"]["dielectric_saturation"] = born_int == 2
        else:
            raise TypeError("elec_model['born_model'] must be a dict or legacy int/string.")

    # Legacy keys inside elec_model for transition compatibility.
    if "dielc_rule" in model:
        out["rel_perm"]["rule"] = model["dielc_rule"]
    if "dielc_diff_mode" in model:
        out["rel_perm"]["differential_mode"] = model["dielc_diff_mode"]
    if "bjeruum_treatment" in model:
        out["DH_model"]["bjeruum_treatment"] = model["bjeruum_treatment"]
    if "born_rel_perm" in model and "eps_r_bulk" in model:
        raise ValueError("Use only one Born permittivity selector in elec_model: born_rel_perm or eps_r_bulk.")
    if "born_rel_perm" in model:
        out["born_model"]["bulk_mode"] = model["born_rel_perm"]
    if "eps_r_bulk" in model:
        out["born_model"]["bulk_mode"] = model["eps_r_bulk"]
    if "born_term_options" in model:
        term = model["born_term_options"]
        if not isinstance(term, dict):
            raise TypeError("Legacy elec_model['born_term_options'] must be a dict.")
        mu_model = out["born_model"]["mu_born_model"]
        if "numerical" in term:
            mu_model["differential_mode"] = "numerical" if _coerce_bool(term["numerical"]) else "analytical"
        if "sum_term" in term:
            mu_model["include_sum_term"] = _coerce_bool(term["sum_term"])
        if "deps_dx_term" in term:
            mu_model["comp_dep_rel_perm"] = _coerce_bool(term["deps_dx_term"])
        if "d_born_mode" in term:
            # Legacy values were 1..5.
            d_old = int(_as_rule_number(term["d_born_mode"], {}))
            legacy_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 3}
            if d_old not in legacy_map:
                raise ValueError("Legacy born_term_options['d_born_mode'] must be in {1,2,3,4,5}.")
            out["born_model"]["d_Born_mode"] = legacy_map[d_old]

    # Canonical coercions.
    out["rel_perm"]["rule"] = _as_rule_number(out["rel_perm"]["rule"], _REL_PERM_RULE_ALIASES)
    out["rel_perm"]["differential_mode"] = _as_rule_number(
        out["rel_perm"]["differential_mode"], _DIFF_MODE_ALIASES
    )
    for key in ("hc_model", "disp_model", "assoc_model", "polar_model"):
        out[key]["dadx_differential_mode"] = _as_rule_number(
            out[key]["dadx_differential_mode"], _DIFF_MODE_ALIASES
        )
    out["DH_model"]["d_ion_mode"] = _resolve_d_ion_mode(out["DH_model"]["d_ion_mode"])
    out["DH_model"]["bjeruum_treatment"] = _coerce_bool(out["DH_model"]["bjeruum_treatment"])
    mu_dh_model = out["DH_model"].get("mu_DH_model", {})
    if not isinstance(mu_dh_model, dict):
        raise TypeError("elec_model['DH_model']['mu_DH_model'] must be a dict.")
    out["DH_model"]["mu_DH_model"] = _deep_update(_CANONICAL_ELEC_MODEL["DH_model"]["mu_DH_model"], mu_dh_model)
    out["DH_model"]["mu_DH_model"]["differential_mode"] = _as_rule_number(
        out["DH_model"]["mu_DH_model"]["differential_mode"], _DIFF_MODE_ALIASES
    )
    out["DH_model"]["mu_DH_model"]["comp_dep_rel_perm"] = _coerce_bool(
        out["DH_model"]["mu_DH_model"]["comp_dep_rel_perm"]
    )
    out["DH_model"]["mu_DH_model"]["include_sum_term"] = _coerce_bool(
        out["DH_model"]["mu_DH_model"]["include_sum_term"]
    )
    out["include_born_model"] = _coerce_bool(out["include_born_model"])

    born = out["born_model"]
    born["d_Born_mode"] = _resolve_d_born_mode(born["d_Born_mode"])
    born["solvation_shell_model"] = _coerce_bool(born["solvation_shell_model"])
    born["dielectric_saturation"] = _coerce_bool(born["dielectric_saturation"])
    born["bulk_mode"] = "solvent" if _resolve_born_bulk_mode(born["bulk_mode"]) == 1 else "mix"

    mu_born_model = born.get("mu_born_model", {})
    if not isinstance(mu_born_model, dict):
        raise TypeError("elec_model['born_model']['mu_born_model'] must be a dict.")
    born["mu_born_model"] = _deep_update(_CANONICAL_ELEC_MODEL["born_model"]["mu_born_model"], mu_born_model)
    born["mu_born_model"]["differential_mode"] = _as_rule_number(
        born["mu_born_model"]["differential_mode"], _DIFF_MODE_ALIASES
    )
    born["mu_born_model"]["comp_dep_rel_perm"] = _coerce_bool(born["mu_born_model"]["comp_dep_rel_perm"])
    born["mu_born_model"]["include_sum_term"] = _coerce_bool(born["mu_born_model"]["include_sum_term"])
    born["mu_born_model"]["comp_dep_delta_d"] = _coerce_bool(born["mu_born_model"]["comp_dep_delta_d"])

    return out


def _flatten_model_to_runtime(model: dict) -> dict:
    rel_perm = model["rel_perm"]
    dh_model = model["DH_model"]
    mu_dh = dh_model["mu_DH_model"]
    born = model["born_model"]
    mu_born = born["mu_born_model"]

    include_born_model = _coerce_bool(model["include_born_model"])
    solvation_shell_model = _coerce_bool(born["solvation_shell_model"])
    dielectric_saturation = _coerce_bool(born["dielectric_saturation"])
    born_bulk_mode = _resolve_born_bulk_mode(born["bulk_mode"])
    mu_dh_diff_mode = _as_rule_number(mu_dh["differential_mode"], _DIFF_MODE_ALIASES)
    mu_born_diff_mode = _as_rule_number(mu_born["differential_mode"], _DIFF_MODE_ALIASES)

    # Transitional legacy runtime projections.
    if not include_born_model:
        legacy_born_model = 0
    elif solvation_shell_model or dielectric_saturation:
        legacy_born_model = 2
    else:
        legacy_born_model = 1
    legacy_radius_map = {0: 1, 1: 2, 2: 3, 3: 5}
    legacy_born_radius_model = int(legacy_radius_map[int(born["d_Born_mode"])])
    if not include_born_model:
        legacy_born_diff_mode = 0
    elif mu_born_diff_mode == 1:
        legacy_born_diff_mode = 1
    elif not _coerce_bool(mu_born["comp_dep_rel_perm"]):
        legacy_born_diff_mode = 3
    elif not _coerce_bool(mu_born["include_sum_term"]):
        legacy_born_diff_mode = 2
    else:
        legacy_born_diff_mode = 0

    return {
        "dielc_rule": int(rel_perm["rule"]),
        "dielc_diff_mode": int(rel_perm["differential_mode"]),
        "hc_dadx_diff_mode": int(_as_rule_number(model["hc_model"]["dadx_differential_mode"], _DIFF_MODE_ALIASES)),
        "disp_dadx_diff_mode": int(_as_rule_number(model["disp_model"]["dadx_differential_mode"], _DIFF_MODE_ALIASES)),
        "assoc_dadx_diff_mode": int(_as_rule_number(model["assoc_model"]["dadx_differential_mode"], _DIFF_MODE_ALIASES)),
        "polar_dadx_diff_mode": int(_as_rule_number(model["polar_model"]["dadx_differential_mode"], _DIFF_MODE_ALIASES)),
        "d_ion_mode": int(_resolve_d_ion_mode(dh_model["d_ion_mode"])),
        "bjeruum_treatment": _coerce_bool(dh_model["bjeruum_treatment"]),
        "mu_DH_diff_mode": int(mu_dh_diff_mode),
        "mu_DH_comp_dep_rel_perm": _coerce_bool(mu_dh["comp_dep_rel_perm"]),
        "mu_DH_include_sum_term": _coerce_bool(mu_dh["include_sum_term"]),
        "include_born_model": include_born_model,
        "d_Born_mode": int(_resolve_d_born_mode(born["d_Born_mode"])),
        "born_solvation_shell_model": solvation_shell_model,
        "born_dielectric_saturation": dielectric_saturation,
        "born_bulk_mode": int(born_bulk_mode),
        "mu_born_diff_mode": int(mu_born_diff_mode),
        "mu_born_comp_dep_rel_perm": _coerce_bool(mu_born["comp_dep_rel_perm"]),
        "mu_born_include_sum_term": _coerce_bool(mu_born["include_sum_term"]),
        "mu_born_comp_dep_delta_d": _coerce_bool(mu_born["comp_dep_delta_d"]),
        # Transitional legacy fields used by older callers/tests.
        "born_model": int(legacy_born_model),
        "born_radius_model": int(legacy_born_radius_model),
        "born_diff_mode": int(legacy_born_diff_mode),
        "born_eps_mode": int(born_bulk_mode),
        "DH_model": 2 if _coerce_bool(dh_model["bjeruum_treatment"]) else 1,
    }


def _resolve_runtime_options(user_options=None) -> dict:
    if user_options is None:
        user_options = {}
    if not isinstance(user_options, dict):
        raise TypeError("user_options must be a dict.")

    allowed = {
        "elec_model",
        "solvated_ion_diameter_mixing_rule",
        "ion_dispersion_mixing_rule",
        # Legacy top-level pass-through keys.
        "bjeruum_treatment",
        "dielc_rule",
        "dielc_diff_mode",
        "born_rel_perm",
        "eps_r_bulk",
        "DH_model",
        "debug",
        "dadt_differential_mode",
    }
    unknown = set(user_options) - allowed
    if unknown:
        raise KeyError(f"Unknown user_options key(s): {sorted(unknown)}")

    model = _normalize_elec_model(user_options.get("elec_model", {}))

    # Legacy top-level compatibility: map into nested model.
    if "dielc_rule" in user_options:
        model["rel_perm"]["rule"] = user_options["dielc_rule"]
    if "dielc_diff_mode" in user_options:
        model["rel_perm"]["differential_mode"] = user_options["dielc_diff_mode"]
    if "bjeruum_treatment" in user_options:
        model["DH_model"]["bjeruum_treatment"] = _coerce_bool(user_options["bjeruum_treatment"])
    if "DH_model" in user_options:
        model["DH_model"]["bjeruum_treatment"] = int(user_options["DH_model"]) == 2

    if "born_rel_perm" in user_options and "eps_r_bulk" in user_options:
        raise ValueError("Use only one Born permittivity selector: born_rel_perm or eps_r_bulk.")
    if "born_rel_perm" in user_options:
        model["born_model"]["bulk_mode"] = user_options["born_rel_perm"]
    elif "eps_r_bulk" in user_options:
        model["born_model"]["bulk_mode"] = user_options["eps_r_bulk"]

    model = _normalize_elec_model(model)
    runtime = _flatten_model_to_runtime(model)
    runtime["debug"] = bool(user_options.get("debug", False))
    runtime["dadt_differential_mode"] = int(_as_rule_number(user_options.get("dadt_differential_mode", "analytical"), _DIFF_MODE_ALIASES))
    if runtime["dadt_differential_mode"] not in (0, 1, 2):
        raise ValueError("dadt_differential_mode must be analytical/numerical/autodiff (0/1/2).")
    runtime["solvated_ion_diameter_mixing_rule"] = _coerce_bool(
        user_options.get("solvated_ion_diameter_mixing_rule", False)
    )
    runtime["ion_dispersion_mixing_rule"] = _coerce_bool(
        user_options.get("ion_dispersion_mixing_rule", True)
    )

    return {
        "runtime": runtime,
        "model": model,
        "preset_key": "dataset_canonical",
        "preset": {},
        "catalog": None,
    }


def _default_species_entry(species_name: str) -> dict:
    comp = _normalize_component(species_name)
    entry = _COMPONENT_DEFAULTS.get(comp)
    if entry is None:
        raise KeyError(f"No default data for species '{species_name}'.")
    resolved = {}
    for key, value in entry.items():
        resolved[key] = value(298.15) if callable(value) else value
    return resolved


def _infer_pure_set_key(components: Iterable[str]) -> str | None:
    neutrals = [comp for comp in components if not comp.endswith("+") and not comp.endswith("-")]
    if len(neutrals) != 1:
        return None
    return _normalize_pure_set_key(neutrals[0])


def _as_composition_array(x, size: int) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1 or x_arr.size != size:
        raise ValueError(f"x must be a 1D array-like vector with length {size}.")
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x contains non-finite values.")
    return x_arr


def _salt_free_neutral_fractions(x, charges) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    z_arr = np.asarray(charges, dtype=float)
    neutral_idx = np.flatnonzero(np.abs(z_arr) <= 1e-12)
    if neutral_idx.size == 0:
        return neutral_idx, np.array([], dtype=float)
    neutral_x = np.clip(x_arr[neutral_idx], 0.0, None)
    total = float(np.sum(neutral_x))
    if total <= 0.0:
        return neutral_idx, np.array([], dtype=float)
    return neutral_idx, neutral_x / total


def _lookup_specific_mixed_rel_perm(
    dataset: dict,
    components: list[str],
    charges,
    x,
    *,
    atol: float = 5.0e-6,
) -> tuple[float | None, str | None]:
    neutral_idx, neutral_sf = _salt_free_neutral_fractions(x, charges)
    if neutral_idx.size < 2 or neutral_sf.size != neutral_idx.size:
        return None, None

    token_fractions: dict[str, float] = {}
    tokens: list[str] = []
    for idx, frac in zip(neutral_idx, neutral_sf):
        token = _solvent_token_for_component(components[int(idx)])
        if token is None:
            return None, None
        tokens.append(token)
        token_fractions[token] = float(frac)

    system_key = _solvent_system_data_key(tokens)
    if not system_key:
        return None, None

    entries = dataset.get("mixed_rel_perm_tables", {}).get(system_key, [])
    if not entries:
        return None, None

    for entry in entries:
        entry_x = entry["x"]
        if set(entry_x) != set(token_fractions):
            continue
        if all(abs(float(entry_x[token]) - float(token_fractions[token])) <= atol for token in token_fractions):
            return float(entry["dielc"]), system_key
    return None, system_key


def _compute_constant_mixed_rel_perm(
    components: list[str],
    charges,
    dielc,
    x,
    mixed_rel_perm: dict[str, dict[str, float]],
) -> float | None:
    neutral_idx, neutral_sf = _salt_free_neutral_fractions(x, charges)
    if neutral_idx.size < 2 or neutral_sf.size != neutral_idx.size:
        return None

    water_pos = None
    for pos, idx in enumerate(neutral_idx):
        if components[idx] == "H2O":
            water_pos = pos
            break
    if water_pos is None:
        return None

    xw_sf = float(neutral_sf[water_pos])
    water_eps = float(dielc[neutral_idx[water_pos]])
    if xw_sf >= 1.0 - 1e-12:
        return water_eps

    x_org = 0.0
    eps_org_num = 0.0
    a_num = 0.0
    b_num = 0.0
    c_num = 0.0
    for pos, idx in enumerate(neutral_idx):
        if pos == water_pos:
            continue
        frac = float(neutral_sf[pos])
        if frac <= 0.0:
            continue
        coeffs = mixed_rel_perm.get(components[idx])
        if coeffs is None:
            return None
        x_org += frac
        eps_org_num += frac * float(dielc[idx])
        a_num += frac * float(coeffs["a"])
        b_num += frac * float(coeffs["b"])
        c_num += frac * float(coeffs["c"])

    if x_org <= 1e-12:
        return water_eps

    eps_org = eps_org_num / x_org
    if xw_sf <= 1e-12:
        return eps_org

    a_eff = a_num / x_org
    b_eff = b_num / x_org
    c_eff = c_num / x_org
    return eps_org + ((a_eff * xw_sf + b_eff) * xw_sf + c_eff) * xw_sf


def _compute_constant_salt_free_weight_avg_rel_perm(
    charges,
    dielc,
    mw,
    x,
) -> float | None:
    neutral_idx, neutral_sf = _salt_free_neutral_fractions(x, charges)
    if neutral_idx.size < 2 or neutral_sf.size != neutral_idx.size:
        return None

    mw_neutral = np.asarray(mw, dtype=float)[neutral_idx]
    if mw_neutral.size != neutral_sf.size or np.any(~np.isfinite(mw_neutral)) or np.any(mw_neutral <= 0.0):
        return None

    dielc_neutral = np.asarray(dielc, dtype=float)[neutral_idx]
    if dielc_neutral.size != neutral_sf.size or np.any(~np.isfinite(dielc_neutral)):
        return None

    mass_weights = neutral_sf * mw_neutral
    total_mass = float(np.sum(mass_weights))
    if total_mass <= 0.0:
        return None
    mass_weights = mass_weights / total_mass
    return float(np.dot(mass_weights, dielc_neutral))


def _apply_constant_mixed_rel_perm_precompute(
    prop_dic: dict,
    dataset: dict,
    components: list[str],
    x,
    rel_perm_rule: int,
) -> None:
    if int(rel_perm_rule) != 0:
        return
    exact_eps, exact_source = _lookup_specific_mixed_rel_perm(
        dataset=dataset,
        components=components,
        charges=prop_dic["z"],
        x=x,
    )
    if exact_eps is not None:
        mixed_eps = float(exact_eps)
        source = "specific"
    else:
        mixed_rel_perm = dataset.get("mixed_rel_perm", {})
        mixed_eps = None
        if mixed_rel_perm:
            mixed_eps = _compute_constant_mixed_rel_perm(
                components=components,
                charges=prop_dic["z"],
                dielc=prop_dic["dielc"],
                x=x,
                mixed_rel_perm=mixed_rel_perm,
            )
        if mixed_eps is None:
            mixed_eps = _compute_constant_salt_free_weight_avg_rel_perm(
                charges=prop_dic["z"],
                dielc=prop_dic["dielc"],
                mw=prop_dic["MW"],
                x=x,
            )
            if mixed_eps is None:
                return
            source = "salt_free_weight_average"
        else:
            source = "empirical"

    neutral_idx, _ = _salt_free_neutral_fractions(x, prop_dic["z"])
    if neutral_idx.size < 2:
        return

    dielc = np.asarray(prop_dic["dielc"], dtype=float).copy()
    dielc[neutral_idx] = float(mixed_eps)
    prop_dic["dielc"] = dielc
    prop_dic["mixed_solvent_rel_perm"] = float(mixed_eps)
    prop_dic["mixed_solvent_rel_perm_applied"] = True
    prop_dic["mixed_solvent_rel_perm_source"] = source
    if exact_source is not None:
        prop_dic["mixed_solvent_rel_perm_dataset"] = exact_source


def _apply_mixed_solvent_ion_sigma(
    prop_dic: dict,
    dataset: dict,
    components: list[str],
    x,
    T: float,
    enabled: bool,
) -> None:
    if not enabled:
        return

    neutral_idx, neutral_sf = _salt_free_neutral_fractions(x, prop_dic["z"])
    if neutral_idx.size < 2 or neutral_sf.size != neutral_idx.size:
        return

    pure_sets = dataset.get("pure_sets", {})
    if not pure_sets:
        raise ValueError(
            f"Dataset '{dataset['dataset_name']}' requires pure/*.csv solvent parameter sets for mixed ion sigma precompute."
        )

    sigma = np.asarray(prop_dic["s"], dtype=float).copy()
    mixed_sigmas: dict[str, float] = {}
    source_weights: dict[str, float] = {}
    for i, comp in enumerate(components):
        if abs(float(prop_dic["z"][i])) <= 1e-12:
            continue
        sigma_mix = 0.0
        for idx, frac in zip(neutral_idx, neutral_sf):
            solvent = components[int(idx)]
            pure_key = _normalize_pure_set_key(solvent)
            sigma_value, source_key = _resolve_component_field_with_source(
                dataset, comp, "s", T, pure_set_key=pure_key
            )
            sigma_mix += float(frac) * float(sigma_value)
            resolved_key = source_key or pure_key
            source_name = f"pure/{resolved_key}.csv"
            source_weights[source_name] = source_weights.get(source_name, 0.0) + float(frac)
        sigma[i] = sigma_mix
        mixed_sigmas[comp] = float(sigma_mix)

    if mixed_sigmas:
        prop_dic["s"] = sigma
        prop_dic["mixed_ion_sigma"] = mixed_sigmas
        prop_dic["mixed_ion_sigma_applied"] = True
        prop_dic["mixed_ion_sigma_sources"] = source_weights


def _apply_mixed_solvent_ion_dispersion(
    prop_dic: dict,
    dataset: dict,
    components: list[str],
    x,
    T: float,
    enabled: bool,
) -> None:
    if not enabled:
        return

    neutral_idx, neutral_sf = _salt_free_neutral_fractions(x, prop_dic["z"])
    if neutral_idx.size < 2 or neutral_sf.size != neutral_idx.size:
        return

    pure_sets = dataset.get("pure_sets", {})
    if not pure_sets:
        raise ValueError(
            f"Dataset '{dataset['dataset_name']}' requires pure/*.csv solvent parameter sets for mixed ion dispersion precompute."
        )

    dispersion = np.asarray(prop_dic["e"], dtype=float).copy()
    mixed_dispersion: dict[str, float] = {}
    source_weights: dict[str, float] = {}
    for i, comp in enumerate(components):
        if abs(float(prop_dic["z"][i])) <= 1e-12:
            continue
        e_mix = 0.0
        for idx, frac in zip(neutral_idx, neutral_sf):
            solvent = components[int(idx)]
            pure_key = _normalize_pure_set_key(solvent)
            e_value, source_key = _resolve_component_field_with_source(
                dataset, comp, "e", T, pure_set_key=pure_key
            )
            e_mix += float(frac) * float(e_value)
            resolved_key = source_key or pure_key
            source_name = f"pure/{resolved_key}.csv"
            source_weights[source_name] = source_weights.get(source_name, 0.0) + float(frac)
        dispersion[i] = e_mix
        mixed_dispersion[comp] = float(e_mix)

    if mixed_dispersion:
        prop_dic["e"] = dispersion
        prop_dic["mixed_ion_dispersion"] = mixed_dispersion
        prop_dic["mixed_ion_dispersion_applied"] = True
        prop_dic["mixed_ion_dispersion_sources"] = source_weights


def molality_to_molefraction(molality, species=None, solvent=None, basis_mass_kg=1.0):
    """Convert salt molality (mol/kg solvent) to species mole-fraction vector."""
    if species is None:
        raise ValueError("species must be provided.")

    species = list(species)
    molality = float(molality)
    basis_mass_kg = float(basis_mass_kg)

    cations = [sp for sp in species if sp.endswith("+")]
    anions = [sp for sp in species if sp.endswith("-")]
    if len(cations) != 1 or len(anions) != 1:
        cations = [sp for sp in species if _default_species_entry(sp).get("z", 0.0) > 0.0]
        anions = [sp for sp in species if _default_species_entry(sp).get("z", 0.0) < 0.0]
    if len(cations) != 1 or len(anions) != 1:
        raise ValueError("Expected exactly one cation and one anion in species list.")

    cation = cations[0]
    anion = anions[0]

    if solvent is None:
        neutrals = [sp for sp in species if _default_species_entry(sp).get("z", 0.0) == 0.0]
        if len(neutrals) != 1:
            raise ValueError("Expected exactly one neutral solvent species when solvent is not specified.")
        solvent = neutrals[0]
    elif solvent not in species:
        raise ValueError(f"Solvent '{solvent}' is not present in species list.")

    z_cat = float(_default_species_entry(cation).get("z", 0.0))
    z_an = float(_default_species_entry(anion).get("z", 0.0))
    if z_cat <= 0.0 or z_an >= 0.0:
        raise ValueError("Invalid cation/anion charges in species list.")

    z_cat_abs = int(round(abs(z_cat)))
    z_an_abs = int(round(abs(z_an)))
    gcd_z = math.gcd(z_cat_abs, z_an_abs)
    v_cat = z_an_abs // gcd_z
    v_an = z_cat_abs // gcd_z

    mw_solvent = float(_default_species_entry(solvent).get("MW", np.nan))
    if not np.isfinite(mw_solvent) or mw_solvent <= 0.0:
        raise ValueError(f"Invalid MW for solvent '{solvent}'.")

    n_solvent = basis_mass_kg / mw_solvent
    n_cation = molality * basis_mass_kg * v_cat
    n_anion = molality * basis_mass_kg * v_an

    n_totals = {sp: 0.0 for sp in species}
    n_totals[solvent] += n_solvent
    n_totals[cation] += n_cation
    n_totals[anion] += n_anion

    total = sum(n_totals.values())
    if total <= 0.0:
        raise ValueError("Computed total moles is non-positive.")

    return np.array([n_totals[sp] / total for sp in species], dtype=float)


def molefraction_to_molality(x, species):
    """Convert mole fractions to molality for 1:1 salt systems."""
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("x must be a 1D array-like vector.")
    if len(species) != x_arr.size:
        raise ValueError("species and x length mismatch.")

    charges = np.asarray([float(_default_species_entry(sp).get("z", 0.0)) for sp in species], dtype=float)
    cation_idx = [i for i, z in enumerate(charges) if z > 0.0]
    solvent_idx = [i for i, z in enumerate(charges) if abs(z) < 1e-12]
    if not cation_idx or not solvent_idx:
        raise ValueError("Need at least one cation and one solvent component.")

    solvent_i = solvent_idx[-1]
    mw_solvent = float(_default_species_entry(species[solvent_i]).get("MW", np.nan))
    if not np.isfinite(mw_solvent) or mw_solvent <= 0.0:
        raise ValueError("Could not resolve solvent MW.")
    if x_arr[solvent_i] <= 0.0:
        raise ValueError("Solvent mole fraction must be > 0 to compute molality.")

    return float(x_arr[cation_idx[0]] / (x_arr[solvent_i] * mw_solvent))


def get_prop_dict(dataset_name: str, species: Iterable[str], x, T: float, user_options: dict | None = None) -> dict:
    """Build a runtime parameter dictionary from a named dataset."""
    dataset = _load_dataset(dataset_name)
    species = list(species)
    components = [_normalize_component(s) for s in species]
    x_arr = _as_composition_array(x, len(components))
    pure_set_key = _infer_pure_set_key(components)

    merged_options = _deep_update(dataset["canonical_user_options"], user_options or {})
    resolved = _resolve_runtime_options(merged_options)
    runtime = resolved["runtime"]

    prop_dic: dict = {}
    for field in BASE_KEYS:
        values = [_resolve_component_field(dataset, comp, field, T, pure_set_key=pure_set_key) for comp in components]
        if field == "assoc_scheme":
            prop_dic[field] = list(values)
        else:
            prop_dic[field] = np.asarray(values, dtype=float)

    for field in OPTIONAL_KEYS:
        values = []
        for comp in components:
            try:
                values.append(_resolve_component_field(dataset, comp, field, T, pure_set_key=pure_set_key))
            except KeyError:
                values.append(0.0)
        prop_dic[field] = np.asarray(values, dtype=float)

    n = len(species)
    k_ij = np.zeros((n, n), dtype=float)
    l_ij = np.zeros((n, n), dtype=float)
    k_hb = np.zeros((n, n), dtype=float)
    for i, c1 in enumerate(components):
        for j, c2 in enumerate(components):
            k_ij[i, j] = _matrix_value(dataset, "k_ij", c1, c2, T)
            l_ij[i, j] = _matrix_value(dataset, "l_ij", c1, c2, T)
            k_hb[i, j] = _matrix_value(dataset, "k_hb", c1, c2, T)

    prop_dic["k_ij"] = k_ij
    prop_dic["l_ij"] = l_ij
    prop_dic["k_hb"] = k_hb

    mixed_rel_perm = dataset.get("mixed_rel_perm", {})
    if mixed_rel_perm:
        prop_dic["mixed_rel_perm_a"] = np.zeros(n, dtype=float)
        prop_dic["mixed_rel_perm_b"] = np.zeros(n, dtype=float)
        prop_dic["mixed_rel_perm_c"] = np.zeros(n, dtype=float)
        prop_dic["mixed_rel_perm_mask"] = np.zeros(n, dtype=int)
        water_index = -1
        for i, comp in enumerate(components):
            if comp == "H2O":
                water_index = i
            coeffs = mixed_rel_perm.get(comp)
            if coeffs is None:
                continue
            prop_dic["mixed_rel_perm_a"][i] = float(coeffs["a"])
            prop_dic["mixed_rel_perm_b"][i] = float(coeffs["b"])
            prop_dic["mixed_rel_perm_c"][i] = float(coeffs["c"])
            prop_dic["mixed_rel_perm_mask"][i] = 1
        prop_dic["mixed_rel_perm_water_index"] = int(water_index)

    _apply_constant_mixed_rel_perm_precompute(
        prop_dic=prop_dic,
        dataset=dataset,
        components=components,
        x=x_arr,
        rel_perm_rule=runtime["dielc_rule"],
    )
    _apply_mixed_solvent_ion_sigma(
        prop_dic=prop_dic,
        dataset=dataset,
        components=components,
        x=x_arr,
        T=T,
        enabled=bool(runtime["solvated_ion_diameter_mixing_rule"]),
    )
    _apply_mixed_solvent_ion_dispersion(
        prop_dic=prop_dic,
        dataset=dataset,
        components=components,
        x=x_arr,
        T=T,
        enabled=bool(runtime["ion_dispersion_mixing_rule"]),
    )

    if np.all(np.abs(prop_dic["z"]) < 1e-12):
        prop_dic["z"] = np.array([])

    prop_dic["elec_model"] = copy.deepcopy(resolved["model"])
    prop_dic["elec_model_dataset"] = dataset_name
    prop_dic["solvated_ion_diameter_mixing_rule"] = bool(runtime["solvated_ion_diameter_mixing_rule"])
    prop_dic["ion_dispersion_mixing_rule"] = bool(runtime["ion_dispersion_mixing_rule"])
    prop_dic["debug"] = bool(runtime["debug"])
    prop_dic["dadt_differential_mode"] = int(runtime["dadt_differential_mode"])
    return prop_dic

