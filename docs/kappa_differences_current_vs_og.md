# Kappa comparison and differences

Scope:
- Reference equations: `C:\Users\Tanner\Documents\git\PC-SAFT\docs\equations.tex`
- Current code: `C:\Users\Tanner\Documents\git\PC-SAFT\pcsaft_electrolyte.cpp`
- Original code: `C:\Users\Tanner\Documents\git\PC-SAFT\pcsaft_electrolyte_og.cpp`

## Reference definition (equations.tex)

Debye-Huckel kappa is defined explicitly as:

\begin{equation}
\kappa=\sqrt{\frac{\rho e^{2}}{k_{B}T\varepsilon_{0}\varepsilon_{r}}\sum_{j}x_{j}z_{j}^{2}}
\end{equation}

## Original C++ (pcsaft_electrolyte_og.cpp)

- Uses the same explicit form as the reference:
  - `kappa = sqrt(den*E_CHRG*E_CHRG/kb/t/(cppargs.dielc*perm_vac)*sum_z2)`
- `sum_z2 = sum_i x[i]*z[i]^2`
- Appears in:
  - Z (ion term)
  - mu_ion
  - ares_ion
  - dadt_ion

## Current C++ (pcsaft_electrolyte.cpp)

- Uses an *implicit* kappa solved by fixed-point iteration:
  - Start with `sum_z2`, compute `kappa`.
  - Compute chi_i(kappa).
  - Update `kappa = sqrt(A * sum_z2_chi / eps_r)` with `sum_z2_chi = sum_i x_i z_i^2 chi_i(kappa)`.
  - Iterate until convergence.
- This means kappa satisfies:
  - `kappa^2 = (A/eps_r) * sum_i x_i z_i^2 * chi_i(kappa)`
- This departs from the explicit reference and the original code.

## Implications

- **Magnitude:** Because `chi_i(kappa) < 1` for finite ion size, `sum_z2_chi < sum_z2`, so the implicit kappa is typically smaller than the explicit kappa.
- **Derivatives:** The implicit definition introduces an extra denominator term when differentiating kappa w.r.t. composition (seen in `pcsaft_electrolyte.cpp` via `1 - 0.5*kappa*D/S`). This term is absent in the original code and the reference equations.
- **Consistency:** Original code matches the reference; current code does not.

## Quick mapping by file/function

Original (explicit):
- `pcsaft_electrolyte_og.cpp`: ion term blocks in `pcsaft_Z_cpp`, `pcsaft_lnfug_cpp`, `pcsaft_ares_cpp`, `pcsaft_dadt_cpp`.

Current (implicit):
- `pcsaft_electrolyte.cpp`: `kappa_chi_mix()` and its callers (`pcsaft_Z_cpp`, `pcsaft_lnfug_cpp`, `pcsaft_ares_cpp`, `pcsaft_dadt_cpp`).

## Notes

- If the goal is to match the reference equations, the original explicit kappa is aligned.
- If the goal is a finite-size corrected kappa, document this as a model change and update the reference equations accordingly.
