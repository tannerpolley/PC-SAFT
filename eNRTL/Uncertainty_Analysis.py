from reduced_hessian import get_reduced_hessian
from scipy.linalg import eigh
import numpy as np
import idaes.core.util.scaling as iscale
import pyomo.environ as pyo

def uncertainty_analysis(m_scaled, estimated_vars, estimated_vars_scaled):
    parameters = {'Description': [], 'Name': [], 'Value': [], 'Uncertainty': [], 'Percent': []}

    active_lbs = {}
    active_ubs = {}
    uncertainty = np.zeros(len(estimated_vars))
    for var, val in m_scaled.ipopt_zL_out.items():
        if val > 1e-1:
            active_lbs[var.name] = val
            # print('Variables with active lower bounds:')
            # print(var.name, val)
    for val in m_scaled.ipopt_zU_out.values():
        if val < -1e-1:
            active_ubs[var.name] = val
            # print('Variables with active upper bounds:')
            # print(var.name, val)
    if len(estimated_vars) > 0 and len(active_lbs) + len(active_ubs) == 0:
        H_red = get_reduced_hessian(m_scaled, estimated_vars_scaled)
        W, V = eigh(H_red)
        inv_red_hess = V @ np.diag(1 / W) @ V.T
        W_value = 1 / W[0]

        if W[0] > 0:
            # print(f"Largest eigenvalue of inverse reduced Hessian: {1 / W[0]}")
            # print("\n" + "Variables with most uncertainty:")
            # for i in np.where(abs(V[:, 0]) > 0.3)[0]:
            #     print(str(i) + ": " + estimated_vars_scaled[i].name)

            # print("==================================================")
            # print("========== Variables with uncertainty ============")
            # print("==================================================")

            def gsf(var):
                return iscale.get_scaling_factor(var, default=1)

            for i, var in enumerate(estimated_vars):
                uncertainty[i] = pyo.sqrt(inv_red_hess[i][i]) / gsf(var)
                # print(f"{var.name}: {var.value:.3e} +/- {uncertainty:.3e}")
                # parameters['Description'].append(var.name)
                # parameters['Name'].append(var.name)
                # parameters['Value'].append(round(var.value, 1))
                # parameters['Uncertainty'].append(str(round(uncertainty, 1)))
                # parameters['Percent'].append(abs(uncertainty / var.value))
        else:
            print("Warning, Hessian is not positive definite at solution!")
            print(f"Most negative eigenvalue of reduced Hessian: {W[0]}")
            print("\n" + "Variables associated:")
            for i in np.where(abs(V[:, 0]) > 0.3)[0]:
                print(str(i) + ": " + estimated_vars_scaled[i].name)
            print("Uncertainties cannot be calculated.")
            for i, var in enumerate(estimated_vars):
                print(f"{var.name}: {var.value:.3e}")
    elif len(estimated_vars) > 0:
        print("Warning, active bounds at solution!")
        print("Uncertainties cannot be calculated.")
        for i, var in enumerate(estimated_vars):
            uncertainty[i] = np.NaN
            print(f"{var.name}: {var.value:.3e}")
        W_value = np.NaN
    else:
        print('Feasibility Problem')
        W_value = np.NaN
        # Feasibility problem
        pass

    for i, var in enumerate(estimated_vars):
        parameters['Description'].append(var.name)
        parameters['Name'].append(var.name)
        parameters['Value'].append(round(var.value, 1))
        parameters['Uncertainty'].append(str(round(uncertainty[i], 1)))
        parameters['Percent'].append(abs(uncertainty[i] / var.value))

    return parameters, W_value
