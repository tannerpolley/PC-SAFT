import numpy as np
from scipy.sparse.linalg import spsolve

from idaes.core.util.scaling import get_jacobian
from idaes.core.util.model_statistics import degrees_of_freedom

def get_reduced_hessian(model, parameters):
    # Assumes a least-squares model has been solved to optimality
    # assert degrees_of_freedom(model) == len(parameters)
    # for val in model.ipopt_zL_out.values(): 
    #     assert val<1
    # for val in model.ipopt_zU_out.values(): 
    #     assert val<1
    jac, nlp = get_jacobian(model, scaled=False)
    n_vars = jac.shape[1]
    param_indices = nlp.get_primal_indices(parameters)
    
    A2 = jac[:, param_indices]

    rest_indices = [i for i in range(n_vars)]

    sorted_indices = param_indices.copy()
    sorted_indices.sort()
    # Go through indices backwards to avoid
    # shifting the list indices when deleting
    # parameter indices
    for idx in sorted_indices[::-1]:
        del rest_indices[idx]
    
    assert len(rest_indices) == n_vars - len(param_indices)
    for idx in param_indices:
        assert idx not in rest_indices

    A1 = jac[:, rest_indices]
    projection_matrix = spsolve(
        A1.tocsc(),
        A2.tocsc()
    )
    if len(param_indices) > 1:
        # Typically there's extreme fill-in
        # If len(param_indices) == 1, then a (dense) ndarray is returned
        projection_matrix = projection_matrix.todense() 
    # import pdb; pdb.set_trace()
    assert nlp.get_obj_factor() == 1
    
    H = nlp.evaluate_hessian_lag()
    H = H.tocsr()
    H11 = H[np.ix_(rest_indices, rest_indices)]
    H21 = H[np.ix_(param_indices, rest_indices)]
    H12 = H[np.ix_(rest_indices, param_indices)]
    H22 = H[np.ix_(param_indices, param_indices)]

    PT = projection_matrix.transpose()
    H_red = PT @ H11 @ projection_matrix - PT @ H12 - H21 @ projection_matrix + H22
    return H_red