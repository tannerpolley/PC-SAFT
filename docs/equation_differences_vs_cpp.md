# PC-SAFT equation vs C++ implementation differences

Scope:
- Reference equations: `C:\Users\Tanner\.codex\skills\epc-saft-equilibrium\references\pcsaft-equations-*.md`
- C++ implementation: `C:\Users\Tanner\Documents\git\PC-SAFT\pcsaft_electrolyte.cpp`

This note documents observed mismatches, omissions, or extra model terms. It is intentionally descriptive; no fixes applied.

Note: Per current usage, polar/dipole terms are disabled and can be ignored for now.

## High-level model term coverage

- **Extra in C++:** Polar (dipole) contribution is implemented (Gross & Vrabec term) in Z, ares, and mu. The reference equations do not include a polar term.
  - C++: `pcsaft_Z_cpp`, `pcsaft_lnfug_cpp`, `pcsaft_ares_cpp` (see `Zpolar`, `ares_polar`, `mu_polar`).
- **Not in C++:** Bjerrum treatment and dissociation-degree equations (alpha, R_i, l_B, K_ip, ion-pairing) are absent from C++.
  - References: Bjerrum/Dissociation sections in residual Helmholtz and chemical potential equations.

## Hard-chain / diameter definition

- **Ion diameter temperature dependence:**
  - Reference: `d_i = sigma_i[1 - 0.12 exp(-3 epsilon_i / kT)]` for all components.
  - C++: for ions, overrides to temperature-independent `d_i = sigma_i*(1-0.12)`.
  - C++ locations: `pcsaft_Z_cpp`, `pcsaft_lnfug_cpp`, `pcsaft_ares_cpp`, `pcsaft_dadt_cpp`.

## Dispersion mixing rule

- **Like-charge dispersion suppression:**
  - Reference: standard `epsilon_ij = sqrt(e_i e_j)(1-k_ij)` for all pairs.
  - C++: for charged systems, `e_ij` is only computed when `z_i*z_j <= 0`; for like-charge pairs it is left as 0 (suppresses dispersion between like ions).
  - C++ locations: `pcsaft_Z_cpp`, `pcsaft_lnfug_cpp`, `pcsaft_ares_cpp`, `pcsaft_dadt_cpp` (look for the `z[i]*z[j] <= 0` guard).

## Debye-Huckel (DH) term

- **Kappa definition differs:**
  - Reference: `kappa = sqrt(A * sum_j x_j z_j^2 / eps_r)`.
  - C++: `kappa` is solved via fixed-point iteration using `sum_z2_chi` (chi-weighted) instead of `sum_z2`.
  - C++ location: `kappa_chi_mix`.

- **Bjerrum modification not implemented:**
  - Reference: alpha_j weights, R_i = max(a_i, l_B), and modified `kappa` for Bjerrum treatment.
  - C++: no alpha, no R_i/l_B, no K_ip / dissociation coupling.

## Born term

- **Two models in C++:**
  - Legacy: matches the simple Born equation with `a_i = d_born` (uses `d_born` if provided; otherwise `d_i`).
  - SSM+DS: implemented but differs from the reference equations.
  - C++ location: `ares_born_legacy`, `ares_born_ssm_ds`, `ares_born_model`.

- **SSM+DS differences vs reference:**
  - Reference: includes two terms with separate dielectric constants (`eps_r` and `eps_r,ion`) and `Delta d_i = (f_min-1)/|z_i| * d_i^Born`.
  - C++: uses a single `factor = (1 - 1/eps_r)` for both terms, no `eps_r,ion` term, and `Delta d_i = f_mix * |z_i| * d_born`.
  - C++: `ares_born_ssm_ds`.

- **Chemical-potential derivative:**
  - Reference: provides analytic `d a_Born / d x_i`.
  - C++: uses a numerical finite-difference derivative for `d ares_born / d x_i`.
  - C++: `pcsaft_lnfug_cpp` (see `daborn_dx` loop).

## Association contribution

- **Delta_ij definition differs:**
  - Reference: `Delta^{AB} = d_ij^3 * g_ij^hs * kappa^{AB} * (exp(eps^{AB}/kT)-1)`.
  - C++: uses `ghs * s_ij^3 * volABij * (exp(eABij/t)-1)` where `volABij` is built from `vol_a` and size mixing.
  - C++ location: association block in `pcsaft_Z_cpp`, `pcsaft_lnfug_cpp`, `pcsaft_ares_cpp`, `pcsaft_dadt_cpp`.

- **Association derivative w.r.t. composition:**
  - Reference (chemical potential): uses `dX/dx_k` at fixed `T, v, x_{i!=k}`.
  - C++: solves `dXA_dx` with respect to `rho_i` (commented as `rho_i = x_i * rho`), then uses `den * x[iA[j]] * dXA_dx` in mu_assoc.
  - C++ location: `dXAdx_find` and `pcsaft_lnfug_cpp` association block.

- **Association form (2B assumption):**
  - C++ includes a comment that only 2B association is implemented in `pcsaft_dadt_cpp`.
  - Reference equations are general over sites.

## Pressure / Compressibility factor

- **Born contribution:**
  - Reference: `Z^{Born} = 0`.
  - C++: `Zborn` is set to 0 in `pcsaft_Z_cpp`.
  - Matches reference.

## Dielectric mixing rules

- **Rule coverage:**
  - C++ implements the new 2025 rule (`eps_solv_mix / (1 + 7.01 * x_ion)`) and uses it by default if `dielc_rule == 0`.
  - Reference includes this new rule.

- **Derivatives:**
  - Reference provides explicit `d eps_r / d x_i` formulas for the three mixing rules.
  - C++ derives `d eps_r / d x_i` programmatically from mass fractions and the chosen rule; appears consistent but should be cross-checked if exact symbolic forms are required.

## Notes for follow-up

- If you want a line-by-line mapping, we can annotate each equation group with exact line numbers and add a checklist of ?implemented vs missing.?
- If you want, I can also generate a diff table (equation -> function/line in C++) for easier review.
