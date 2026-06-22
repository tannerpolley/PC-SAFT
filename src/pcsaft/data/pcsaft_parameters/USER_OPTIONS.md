# User Options

Each dataset `user_options.json` in this folder is treated as a sparse override file.
The loader deep-merges it onto the package defaults defined in
`src/pcsaft/parameters.py`, so default-valued entries should be omitted.

## Package defaults

```json
{
  "debug": false,
  "solvated_ion_diameter_mixing_rule": false,
  "ion_dispersion_mixing_rule": true,
  "elec_model": {
    "rel_perm": {
      "rule": 1,
      "differential_mode": "analytical"
    },
    "hc_model": {
      "dadx_differential_mode": "analytical"
    },
    "disp_model": {
      "dadx_differential_mode": "analytical"
    },
    "assoc_model": {
      "dadx_differential_mode": "analytical"
    },
    "polar_model": {
      "dadx_differential_mode": "analytical"
    },
    "DH_model": {
      "d_ion_mode": 1,
      "bjeruum_treatment": false,
      "mu_DH_model": {
        "differential_mode": "analytical",
        "comp_dep_rel_perm": true,
        "include_sum_term": true
      }
    },
    "include_born_model": true,
    "born_model": {
      "d_Born_mode": 0,
      "solvation_shell_model": false,
      "dielectric_saturation": false,
      "bulk_mode": "mix",
      "mu_born_model": {
        "differential_mode": "analytical",
        "comp_dep_rel_perm": true,
        "include_sum_term": true,
        "comp_dep_delta_d": false
      }
    }
  }
}
```

## Supported keys

- Top level: `debug`, `solvated_ion_diameter_mixing_rule`, `ion_dispersion_mixing_rule`, `elec_model`
- `elec_model.rel_perm.rule`: default `1`; accepts integers and aliases such as `constant`, `linear`, `combined`, `empirical`
- `dadt_differential_mode`: default `analytical`; accepts `analytical`, `numerical`, or `autodiff`
- `elec_model.rel_perm.differential_mode`: default `analytical`; accepts `analytical`, `numerical`, or `autodiff`
- `elec_model.hc_model.dadx_differential_mode`: default `analytical`
- `elec_model.disp_model.dadx_differential_mode`: default `analytical`
- `elec_model.assoc_model.dadx_differential_mode`: default `analytical`
- `elec_model.polar_model.dadx_differential_mode`: default `analytical`
- `elec_model.DH_model.d_ion_mode`: default `1`; accepts `0`, `1`, `2` or `t_indep`, `t_dep_1`, `t_dep_2`
- `elec_model.DH_model.bjeruum_treatment`: default `false`
- `elec_model.DH_model.mu_DH_model.differential_mode`: default `analytical`; accepts `analytical`, `numerical`, or `autodiff`
- `elec_model.DH_model.mu_DH_model.comp_dep_rel_perm`: default `true`
- `elec_model.DH_model.mu_DH_model.include_sum_term`: default `true`
- `elec_model.include_born_model`: default `true`
- `elec_model.born_model.d_Born_mode`: default `0`; accepts `0`, `1`, `2`, `3` or `t_indep`, `t_dep_1`, `t_dep_2`, `fitted_param`
- `elec_model.born_model.solvation_shell_model`: default `false`
- `elec_model.born_model.dielectric_saturation`: default `false`
- `elec_model.born_model.bulk_mode`: default `mix`; accepts `mix` or `solvent`
- `elec_model.born_model.mu_born_model.differential_mode`: default `analytical`; accepts `analytical`, `numerical`, or `autodiff`
- `elec_model.born_model.mu_born_model.comp_dep_rel_perm`: default `true`
- `elec_model.born_model.mu_born_model.include_sum_term`: default `true`
- `elec_model.born_model.mu_born_model.comp_dep_delta_d`: default `false`

## Minimal examples

Disable the Born contribution:

```json
{
  "elec_model": {
    "include_born_model": false
  }
}
```

Switch to constant dielectric mixing and disable Born:

```json
{
  "elec_model": {
    "rel_perm": {
      "rule": "constant"
    },
    "include_born_model": false
  }
}
```

Use the Figiel/Khudaida-style non-default Born settings:

```json
{
  "elec_model": {
    "rel_perm": {
      "rule": "empirical",
      "differential_mode": "numerical"
    },
    "born_model": {
      "d_Born_mode": 3,
      "solvation_shell_model": true,
      "dielectric_saturation": true,
      "mu_born_model": {
        "differential_mode": "numerical",
        "comp_dep_delta_d": true
      }
    }
  }
}
```

