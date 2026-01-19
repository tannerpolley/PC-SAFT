from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


class InputError(Exception): ...
class SolutionError(Exception): ...


def pcsaft_den(t: float, p: float, x: np.ndarray, params: Dict[str, Any], phase: str = "liq") -> float: ...
def pcsaft_p(t: float, p_or_rho: float, x: np.ndarray, params: Dict[str, Any], phase: str = "liq", input: str = "p") -> float: ...
def pcsaft_Z(t: float, p_or_rho: float, x: np.ndarray, params: Dict[str, Any], phase: str = "liq", input: str = "p") -> float: ...

def pcsaft_lnfugcoef(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any],
    phase: str = "liq",
    input: str = "p",
) -> np.ndarray: ...

def pcsaft_fugcoef(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any],
    phase: str = "liq",
    input: str = "p",
) -> np.ndarray: ...

def pcsaft_lnfugcoef_inf_dil(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any],
    phase: str = "liq",
    input: str = "p",
    eps: float = ...,
) -> np.ndarray: ...

def pcsaft_fugcoef_inf_dil(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any],
    phase: str = "liq",
    input: str = "p",
    eps: float = ...,
) -> np.ndarray: ...

def pcsaft_actcoeff(
    t: float,
    p_or_rho: float,
    x: np.ndarray,
    params: Dict[str, Any],
    phase: str = "liq",
    input: str = "p",
    eps: float = ...,
) -> np.ndarray: ...

def pcsaft_hres(t: float, p_or_rho: float, x: np.ndarray, params: Dict[str, Any], phase: str = "liq", input: str = "p") -> float: ...
def pcsaft_sres(t: float, p_or_rho: float, x: np.ndarray, params: Dict[str, Any], phase: str = "liq", input: str = "p") -> float: ...
def pcsaft_gres(t: float, p_or_rho: float, x: np.ndarray, params: Dict[str, Any], phase: str = "liq", input: str = "p") -> float: ...
def pcsaft_ares(t: float, p_or_rho: float, x: np.ndarray, params: Dict[str, Any], phase: str = "liq", input: str = "p") -> float: ...
def pcsaft_dadt(t: float, p_or_rho: float, x: np.ndarray, params: Dict[str, Any], phase: str = "liq", input: str = "p") -> float: ...

def pcsaft_osmoticC(t: float, p_or_rho: float, x: np.ndarray, params: Dict[str, Any], phase: str = "liq", input: str = "p") -> float: ...
def pcsaft_cp(t: float, p_or_rho: float, aly_lee_params: Sequence[float], x: np.ndarray, params: Dict[str, Any], phase: str = "liq", input: str = "p") -> float: ...
def pcsaft_Hvap(t: float, x: np.ndarray, params: Dict[str, Any], p_guess: float | None = ...) -> Sequence[float]: ...

def flashTQ(t: float, q: float, x: np.ndarray, params: Dict[str, Any], p_guess: float | None = ...) -> Sequence[float]: ...
def flashPQ(p: float, q: float, x: np.ndarray, params: Dict[str, Any], t_guess: float | None = ...) -> Sequence[float]: ...

def aly_lee(t: float, params: Sequence[float]) -> float: ...
def dielc_water(t: float) -> float: ...
