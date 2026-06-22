# Differences: equations.tex vs current vs original C++ (excluding polar)

Scope:
- Reference equations: `C:\Users\Tanner\Documents\git\PC-SAFT\docs\equations.tex`
- Current code: `C:\Users\Tanner\Documents\git\PC-SAFT\pcsaft_electrolyte.cpp`
- Original code: `C:\Users\Tanner\Documents\git\PC-SAFT\pcsaft_electrolyte_og.cpp`

Notes:
- Polar/dipole terms intentionally ignored.
- Born term exists only in current C++; OG has none.
- Dielectric mixing (dielc_mix + dielc_mix_dx) exists only in current C++.

## Kappa / Debye-Huckel (DH) term

- **Reference:** explicit kappa using ionic strength.
- **OG C++:** explicit kappa using sum(x z^2), matches reference.
- **Current C++ (after changes):** now matches OG/reference for kappa and DH expressions; uses `dielc_mix` in place of constant `cppargs.dielc`.

## DH contributions (Z, ares, mu)

- **Reference:** uses `chi_i` from kappa and explicit DH forms.
- **OG C++:** matches reference for Z_DH, ares_DH, mu_DH (uses `summ1/summ2` form for mu_ion).
- **Current C++:** now matches OG for formulas; only difference is dielectric constant source (`dielc_mix`).

## Association contribution (Delta_ij)

- **Reference:** `Delta^{AB} = d_ij^3 * g_ij^{hs} * [exp(eps^{AB}/kT)-1]`.
- **OG C++:** uses `ghs * s_ij^3 * volABij * (exp(eABij/t)-1)`.
- **Current C++:** same as OG.

## Dispersion mixing rules

- **Reference:** uses `sigma_ij = 0.5*(sigma_i+sigma_j)*(1-l_ij)` and piecewise `epsilon_ij` (zero for like-charge pairs).
- **OG C++:** uses `s_ij` with `l_ij` and suppresses `e_ij` for like-charge pairs (`z_i*z_j>0`).
- **Current C++:** same as OG.

## Ion diameter

- **Reference:** includes explicit ion diameter expression `d_ion = sigma_ion*(1-0.12)`.
- **OG C++:** implements temperature-independent diameter for ions.
- **Current C++:** same as OG.

## Born term

- **Reference:** Born term equations exist (legacy + SSM/DS forms).
- **OG C++:** no Born term.
- **Current C++:** includes Born models and derivatives.

## Dielectric constant

- **Reference:** provides multiple mixing rules (mole, mass, combo, new 2025 rule).
- **OG C++:** uses constant `cppargs.dielc`.
- **Current C++:** implements `dielc_mix`/`dielc_mix_dx` with rule selection.

## Summary of current vs OG

- **Same:** hard-chain, dispersion, association, DH/ion math (after kappa revert), and all non-polar non-Born terms.
- **Different:** dielectric mixing functions and Born term (current only).
- **Ignored:** polar/dipole term (present in both but not active).
