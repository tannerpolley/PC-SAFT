from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


class InputError(Exception): ...
class SolutionError(Exception): ...


def get_prop_dict(
    species: Sequence[str],
    t: float,
    user_params: Dict[str, Any] | None = ...,
) -> Dict[str, Any]: ...

def validate_species_params(
    species: Sequence[str],
    user_params: Dict[str, Any] | None = ...,
) -> Dict[str, Any]: ...

def pcsaft_den(
    t: float,
    p: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
) -> float: ...

def pcsaft_p(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
) -> float: ...

def pcsaft_Z(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> float: ...

def pcsaft_Z_contrib(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> Dict[str, float]: ...

def pcsaft_lnfugcoef(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> np.ndarray: ...

def pcsaft_fugcoef(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> np.ndarray: ...

def pcsaft_lnfugcoef_inf_dil(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
    eps: float = 0.0,
    ion_list: Sequence[str] | None = ...,
) -> np.ndarray | float: ...

def pcsaft_fugcoef_inf_dil(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
    eps: float = 0.0,
    ion_list: Sequence[str] | None = ...,
) -> np.ndarray | float: ...

def pcsaft_gsolv(
    t: float,
    p: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    dielc_rule: int | str | None = ...,
    ion_list: Sequence[str] | None = ...,
) -> np.ndarray | float: ...

def pcsaft_gtransfer(
    t: float,
    p: float,
    x1: np.ndarray,
    x2: np.ndarray,
    params1: Dict[str, Any] | None = ...,
    params2: Dict[str, Any] | None = ...,
    species1: Sequence[str] | None = ...,
    species2: Sequence[str] | None = ...,
    user_params1: Dict[str, Any] | None = ...,
    user_params2: Dict[str, Any] | None = ...,
    dielc_rule1: int | str | None = ...,
    dielc_rule2: int | str | None = ...,
    ion_list: Sequence[str] | None = ...,
) -> np.ndarray | float: ...

def pcsaft_actcoeff(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
    eps: float = 0.0,
) -> np.ndarray: ...

def pcsaft_miac(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
    eps: float = 0.0,
) -> Dict[str, float]: ...

def pcsaft_miac_m(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
    eps: float = 0.0,
) -> Dict[str, float]: ...

def pcsaft_mu_res_contrib(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> Dict[str, Any]: ...

def pcsaft_mures(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> Dict[str, Any]: ...

def pcsaft_ares_contrib(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> Dict[str, float]: ...

def pcsaft_ion_dh_debug(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> Dict[str, Any]: ...

def pcsaft_debug_dielc(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> Dict[str, float]: ...

def pcsaft_debug_bjerrum(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> Dict[str, float]: ...

def pcsaft_salt_molality_to_ionic_x(
    t: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
) -> np.ndarray: ...

def pcsaft_hres(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> float: ...
def pcsaft_sres(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> float: ...
def pcsaft_gres(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> float: ...
def pcsaft_ares(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
    return_contributions: bool = ...,
) -> float | Tuple[float, Dict[str, float]]: ...
def pcsaft_dadt(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> float: ...

def pcsaft_osmoticC(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> float: ...
def pcsaft_cp(
    t: float,
    p_or_rho: float,
    aly_lee_params: Sequence[float],
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    input: str = "p",
) -> float: ...
def pcsaft_Hvap(
    t: float,
    x: np.ndarray,
    params: Dict[str, Any] | None = ...,
    species: Sequence[str] | None = ...,
    user_params: Dict[str, Any] | None = ...,
    phase: str = "liq", dielc_rule: int | str | None = ...,
    p_guess: float | None = ...,
) -> Sequence[float]: ...

def flashTQ(t: float, q: float, x: np.ndarray, params: Dict[str, Any] | None = ..., species: Sequence[str] | None = ..., user_params: Dict[str, Any] | None = ..., phase: str = "liq", dielc_rule: int | str | None = ..., p_guess: float | None = ...) -> Sequence[float]: ...
def flashPQ(p: float, q: float, x: np.ndarray, params: Dict[str, Any] | None = ..., species: Sequence[str] | None = ..., user_params: Dict[str, Any] | None = ..., phase: str = "liq", dielc_rule: int | str | None = ..., t_guess: float | None = ...) -> Sequence[float]: ...

def aly_lee(t: float, params: Sequence[float]) -> float: ...
def dielc_water(t: float) -> float: ...
