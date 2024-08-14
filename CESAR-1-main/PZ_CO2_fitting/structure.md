## Packages


### Python Packages
 - `import pandas as pd`
 - `import numpy as np`
 - `import matplotlib.pyplot as plt`

### Pyomo Packages
- `pyomo`
  - `environ as pyo`
  - `common.fileutils this_file_dir`

### Idaes Packages
- `idaes`
  - `logger as idaeslog`
  - `core`
    - `util.model_statistics import degrees of freedom`
    - `util.math import smooth_abs`
    - `solvers import get_solver`
    - `util.scaling as iscale`

  - `models.properties.modular_properties`
    - `base.generic_property import GenericParameterBlock`
    - `pure import NIST`

### eNRTL file functions
- `get_prop_dict`
- `create_heat_capacity_no_inherent_rxns`
- `initialize_inherent_reactions`



## Pyomo Model 
- Setup optional arguments for the solver
```
optarg = {
    # 'bound_push' : 1e-22,
    'nlp_scaling_method': 'user-scaling',
    'linear_solver': 'ma57',
    'OF_ma57_automatic_scaling': 'yes',
    'max_iter': 500,
    'tol': 1e-8,
    'constr_viol_tol': 1e-8,
    'halt_on_ampl_error': 'no',
    # 'mu_strategy': 'monotone',
}
```
- ### Build Model `m = pyo.ConcreteModel()`
- ### Create and Scale the parameters 
  - `create_and_scale_params(m)` (defined in fitting file)
  - #### Create Parameters
    - Define reaction combinations dictionary 
    ```
    rxn_combinations = {
        "PZ_bicarbonate_formation": {
            "bicarbonate_formation": 1,
            "PZ_protonation": 1
        },
        "PZ_carbamate_formation_combo": {
            "PZ_carbamate_formation": 1,
            "PZ_protonation": 1
        },
        "PZ_carbamate_proton_transfer": {
            "PZ_carbamate_protonation": 1,
            "PZ_protonation": -1
        },
        "PZ_bicarbamate_formation_combo": {
            "PZ_bicarbamate_formation": 1,
            "PZ_protonation": 1
        }
    }
    
    ```
    - These reactions should be defined in the configuration dictionary found in the MEA_eNRTL file 
    - This reaction combination dictionary is reducing the number of total reactions happening in the system 
    and this is the systemitic approach to it rather than explicitly defining the system with only 
    two reactions
    - Create configuration dictionary for the chosen species and specify reactions 
    ```
    config = get_prop_dict(["H2O", "PZ", "CO2"], 
                           excluded_rxns=["H2O_autoionization", 
                                          "carbonate_formation"],
                           rxn_combinations=rxn_combinations)
    ```
    - The `get_prop_dict` function comes from the `eNRTL` file for the corresponding solvent (Cesar 1 or MEA for example)
    - generate params object from the idaes GenericParameterBlock `params = m.params = GenericParameterBlock(**config)`
  - #### Scale Parameters
    - Set scaling for all variables:
    ```
    scaling_factor_flow_mol = 1 / 100
    params.set_default_scaling("enth_mol_phase", 3e-4)
    params.set_default_scaling("pressure", 1e-5)
    params.set_default_scaling("temperature", 1)
    params.set_default_scaling("flow_mol", scaling_factor_flow_mol)
    params.set_default_scaling("flow_mol_phase", scaling_factor_flow_mol)

    params.set_default_scaling("flow_mass_phase", scaling_factor_flow_mol / 18e-3)  # MW mixture ~= 24 g/Mol
    params.set_default_scaling("dens_mol_phase", 1 / 18000)
    params.set_default_scaling("visc_d_phase", 700)
    params.set_default_scaling("log_k_eq", 1)
    ```
    - Set reaction scaling factors 
    ```
    inherent_rxn_scaling_factors = {
          "PZ_bicarbonate_formation": 5e3,
          "PZ_carbamate_formation_combo": 5e2,
          "PZ_carbamate_proton_transfer": 5e2,
          "PZ_bicarbamate_formation_combo": 5e3,
      }
    ```
    - Setup scaling factors for all mole fractions including true species 
    ```    
    for comp, sf_x in mole_frac_scaling_factors.items():
        params.set_default_scaling("mole_frac_comp", sf_x, index=comp)
        params.set_default_scaling("mole_frac_phase_comp", sf_x, index=("Liq", comp))
        params.set_default_scaling("flow_mol_phase_comp",sf_x * scaling_factor_flow_mol, index=("Liq", comp))
    ```
    - Setup scaling factors for reaction extent in a loop 
    ```
    for rxn, sf_xi in inherent_rxn_scaling_factors.items():
          params.set_default_scaling("apparent_inherent_reaction_extent", scaling_factor_flow_mol*sf_xi, index=rxn)
    ```
    - Use iscale to set scaling factors for specific eNRTL parameters 
    ```
    iscale.set_scaling_factor(m.params.Liq.alpha, 1) 
    iscale.set_scaling_factor(m.params.Liq.tau_A, 1) # Reminder that it's well-scaled
    iscale.set_scaling_factor(m.params.Liq.tau_B, 1/300)
    ```
    - Use iscale to set scaling factors for specific chemical equilibrium parameters 
    ```
    for rxn_name in inherent_rxn_scaling_factors.keys():
        rxn_obj = getattr(m.params, "reaction_"+rxn_name)
        iscale.set_scaling_factor(rxn_obj.k_eq_coeff_1, 1)
        iscale.set_scaling_factor(rxn_obj.k_eq_coeff_2, 1/300)
        iscale.set_scaling_factor(rxn_obj.k_eq_coeff_3, 1)
        iscale.set_scaling_factor(rxn_obj.k_eq_coeff_4, 300)
    ```

- ### Unload data and setup objective functions
  - The objective function expressions are defined here first and are added onto for each dataset
  - #### Partial Pressure CO2 fitting
    - Unload data into data frame 
    ```
    df_hillard_loading = pd.read_csv(os.sep.join([data, "hillard_PZ_loading_concatenated.csv"]), index_col=None)
    ```
    - Get length of the df and set up the data as a pyomo parameter 
    ```
    n_data_hillard_loading = len(df_hillard_loading["temperature"])
    m.hillard_loading = m.params.build_state_block(range(n_data_hillard_loading), defined_state=True)
    ```
    - Iterate through each row and load the right data into the newly made parameter and setup the objective function
    ```
    for i, row in df_hillard_loading.iterrows():
        molality = row["PZ_molality"]
        n_PZ = molality
        n_H2O = 1 / 0.01802
        n_CO2 = 2 * n_PZ * row["CO2_loading"]
        n_tot = n_PZ + n_H2O + n_CO2
        m.hillard_loading[i].flow_mol.fix(100)
        m.hillard_loading[i].mole_frac_comp["H2O"].fix(n_H2O / n_tot)
        m.hillard_loading[i].mole_frac_comp["PZ"].fix(n_PZ / n_tot)
        m.hillard_loading[i].mole_frac_comp["CO2"].fix(n_CO2 / n_tot)
        m.hillard_loading[i].pressure.fix(310264 + 101300)
        m.hillard_loading[i].temperature.fix(row["temperature"] + 273.15)  # Temperature in C
        obj_least_squares_expr += (loss_least_squares(m.hillard_loading[i].log_fug_phase_comp["Liq", "CO2"]
                                                      - pyo.log(row["CO2_partial_pressure"] * 1e3)))  # Pressure in kPa
        obj_abs_expr += (loss_abs(m.hillard_loading[i].log_fug_phase_comp["Liq", "CO2"]
                - pyo.log(row["CO2_partial_pressure"] * 1e3)))  # Pressure in kPa
    ```
    - Can do the same step for different data sets (Hillard, Dugas, etc...)
  - #### Enthalpy of Absorption Fitting
    - Similar but more complicated process to setup enthalpy of absorption fitting
  - #### Speciation fitting
    - Similar but more complicated process to setup speciation fitting

  - Use iscale to calculate scaling factors 
  ```
  iscale.calculate_scaling_factors(m)
  ```
  - Initialize the inherent reaction for each data set and then initialize the parameter block for each data set 
  ```
  initialize_inherent_reactions(m.hillard_loading)
  m.hillard_loading.initialize(hold_state=False, outlvl=init_outlevel, optarg=optarg)
  ```
  - The initialize inherent reactions function comes from the eNRTL file
  - Calculate scaling factors again 
  ```
  iscale.calculate_scaling_factors(m)
  ```
- ### Get estimated parameters 
  - `estimated_vars = get_estimated_params(m)` (defined in fitting file)
  - This function sets up a empty list of parameters and appends each relevant parameters estimated value into the list
    - Appends estimated equilirbium constant parameters for each reaction 
    `param_list.append(rxn_obj.k_eq_coeff_1)`
    - Appends estimated eNRTL parameters for the appropriate combinations (solvent, ion) 
    `param_list.append(m.params.Liq.tau_A[solvent, idx])`
  - Include these estimated in the objective function 
  ```
  for var in estimated_vars:
      obj_least_squares_expr += 0.01 * loss_least_squares((var - var.value) * iscale.get_scaling_factor(var))
  ```
- ### Setup of Objective Function, Transformation, and Solver
  - Load objetive functions expressions into the model
  ```
  m.obj_least_squares = pyo.Objective(expr=obj_least_squares_expr)
  m.obj_abs = pyo.Objective(expr=obj_abs_expr)
  m.obj_abs.deactivate()
  ```
  - Scale the model with TransformationFactory (from Pyomo)
  ```
  m_scaled = pyo.TransformationFactory('core.scale_model').create_using(m, rename=False)
  ```
  - Setup and get solver
  ```
  optarg.pop("nlp_scaling_method", None)  # Scaled model doesn't need user scaling
  optarg["max_iter"] = 500
  solver = get_solver("ipopt",options=optarg)
  ```
  - Rescale the estimated parameters
  ```
  estimated_vars_scaled = get_estimated_params(m_scaled)
  for var in estimated_vars_scaled:
      var.unfix()
  ```
- ### Solve the model
  - Run the solver
  ```
  res = solver.solve(m_scaled, tee=True)
  pyo.assert_optimal_termination(res)
  pyo.TransformationFactory('core.scale_model').propagate_solution(m_scaled, m)
  ```
  
## Analysis

