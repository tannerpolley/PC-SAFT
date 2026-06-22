import numpy as np
from pcsaft import pcsaft_lnfugcoef


def main():
    # Example using built-in species parameters
    species = ['Methanol', 'Li+']
    x = np.asarray([0.95, 0.05])
    t = 298.15
    p = 101325.0
    lnphi = pcsaft_lnfugcoef(t, p, x, species=species, phase='liq')
    print('ln(phi) species list:', lnphi)

    # Example using custom user_params
    user_params = {
        'MySolvent': {
            'MW': 0.050,
            'm': 2.5,
            's': 3.2,
            'e': 200.0,
            'e_assoc': 0.0,
            'vol_a': 0.0,
            'assoc_scheme': None,
            'dipm': 0.0,
            'dip_num': 1,
            'z': 0.0,
            'dielc': 30.0,
        }
    }
    species = ['MySolvent']
    x = np.asarray([1.0])
    lnphi = pcsaft_lnfugcoef(t, p, x, species=species, user_params=user_params)
    print('ln(phi) custom component:', lnphi[0])


if __name__ == '__main__':
    main()
