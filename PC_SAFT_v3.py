import numpy as np
from scipy.optimize import root


class PCSAFT_v3:
    a_ni = np.array([[0.9105631445, -0.3084016918, -0.0906148351],
                     [0.6361281449, 0.1860531159, 0.4527842806],
                     [2.6861347891, -2.5030047259, 0.5962700728],
                     [-26.547362491, 21.419793629, -1.7241829131],
                     [97.759208784, -65.255885330, -4.1302112531],
                     [-159.59154087, 83.318680481, 13.776631870],
                     [91.297774084, -33.746922930, -8.6728470368]])
    a_ni = a_ni.T

    b_ni = np.array([[0.7240946941, -0.5755498075, 0.0976883116],
                     [2.2382791861, 0.6995095521, -0.2557574982],
                     [-4.0025849485, 3.8925673390, -9.1558561530],
                     [-21.003576815, -17.215471648, 20.642075974],
                     [26.855641363, 192.67226447, -38.804430052],
                     [206.55133841, -161.82646165, 93.626774077],
                     [-355.60235612, -165.20769346, -29.666905585]])
    b_ni = b_ni.T

    kb = 1.380649e-23  # J/K
    N_A = 6.0221e23  # 1/mol
    R = 8.314  # J/mol-K
    π = np.pi

    def __init__(self, T, z, prop_dic, phase='liquid', η=None, P_sys=None):

        m = prop_dic['m']
        σ = prop_dic['s']
        ϵ_k = prop_dic['e']
        κ_AB = prop_dic['vol_a']
        ϵ_AB_k = prop_dic['e_assoc']
        k_ij = prop_dic['k_ij']

        # Parameters
        self.T = T
        self.T_og = T
        self.z = z
        self.z_og = z
        self.m = m
        self.k = len(σ)
        self.σ = σ
        self.ϵ_k = ϵ_k
        self.κ_AB = κ_AB
        self.ϵ_AB_k = ϵ_AB_k
        self.phase = phase
        self.k = len(z)
        k = self.k

        # --------------------------------------- Intermediates --------------------------------------- #

        self.σ_ij = np.array([[1 / 2 * (σ[i] + σ[j]) for j in range(k)] for i in range(k)])
        self.ϵ_ij = np.array([[(ϵ_k[i] * ϵ_k[j]) ** (1 / 2) * (1 - k_ij[i][j]) for j in range(k)] for i in range(k)])
        self.κ_AB_ij = np.array(
            [[(κ_AB[i] * κ_AB[j]) ** (1 / 2) * ((σ[i] * σ[j]) / (1 / 2 * (σ[i] * σ[j]))) ** 3 for j in range(k)] for i
             in range(k)])
        self.ϵ_AB_ij = np.array([[(ϵ_AB_k[i] + ϵ_AB_k[j]) / 2 for j in range(k)] for i in range(k)])

        if P_sys is not None:
            self.P_sys = P_sys
            self.f_find_η()
        elif P_sys is None and η is not None:
            self.η = η
            self.ρ = self.f_ρ(z=z, T=T, η=η)
        elif P_sys is None and η is None:
            if phase == 'liquid':
                self.η = .4
            elif phase == 'vapor':
                self.η = .01
            print('''
            Warning, a default η had to be defined based on the given phase 
            since so system pressure was given to iteratively find η
            ''')

    def f_m_bar(self, z=None):

        if z is None:
            z = self.z
        m = self.m

        return sum(z * m)

    def f_d(self, T=None):
        if T is None:
            T = self.T
        σ = self.σ
        ϵ_k = self.ϵ_k

        return σ * (1 - .12 * np.exp(-3 * ϵ_k / T))

    def f_ρ(self, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η

        d = self.f_d(T=T)
        m = self.m
        k = self.k
        return 6 / self.π * η * (sum([z[i] * m[i] * d[i] ** 3 for i in range(k)])) ** (-1)

    def f_v(self, ρ=None):

        if ρ is None:
            ρ = self.ρ

        return self.N_A * 10 ** -30 / ρ

    def f_ξ(self, z=None, T=None, ρ=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if ρ is None:
            ρ = self.ρ

        d = self.f_d(T=T)
        m = self.m
        return np.array([self.π / 6 * ρ * np.sum([z[i] * m[i] * d[i] ** n for i in range(self.k)]) for n in range(4)])

    def f_g_hs_ij(self, z=None, T=None, ρ=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if ρ is None:
            ρ = self.ρ

        d = self.f_d(T=T)
        ξ = self.f_ξ(z=z, T=T, ρ=ρ)

        return np.array([[(1 / (1 - ξ[3])) +
                          ((d[i] * d[j] / (d[i] + d[j])) * 3 * ξ[2] / (1 - ξ[3]) ** 2) +
                          ((d[i] * d[j] / (d[i] + d[j])) ** 2 * 2 * ξ[2] ** 2 / (1 - ξ[3]) ** 3)
                          for j in range(self.k)]
                         for i in range(self.k)])

    def f_d_ij(self, T=None):
        if T is None:
            T = self.T

        d = self.f_d(T=T)
        return np.array([[1 / 2 * (d[i] + d[j]) for j in range(self.k)] for i in range(self.k)])

    def f_Δ_AB_ij(self, z=None, T=None, ρ=None):
        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if ρ is None:
            ρ = self.ρ

        d_ij = self.f_d_ij(T=T)
        g_hs_ij = self.f_g_hs_ij(z=z, T=T, ρ=ρ)
        return np.array([[d_ij[i][j] ** 3 * g_hs_ij[i][j] * self.κ_AB_ij[i][j] * (np.exp(self.ϵ_AB_ij[i][j] / T) - 1)
                          for j in range(self.k)] for i in range(self.k)])

    def f_a_hs(self, z=None, T=None, ρ=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if ρ is None:
            ρ = self.ρ

        ξ = self.f_ξ(z=z, T=T, ρ=ρ)
        return 1 / ξ[0] * (3 * ξ[1] * ξ[2] / (1 - ξ[3]) + ξ[2] ** 3 / (ξ[3] * (1 - ξ[3]) ** 2) + (
                ξ[2] ** 3 / ξ[3] ** 2 - ξ[0]) * np.log(1 - ξ[3]))

    def f_a_hc(self, z=None, T=None, ρ=None):
        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if ρ is None:
            ρ = self.ρ

        k = self.k
        m = self.m
        m_bar = self.f_m_bar(z=z)
        g_hs_ij = self.f_g_hs_ij(z=z, T=T, ρ=ρ)
        a_hs = self.f_a_hs(z=z, T=T, ρ=ρ)

        return m_bar * a_hs - sum([z[i] * (m[i] - 1) * np.log(g_hs_ij[i][i]) for i in range(k)])

    def f_a_disp(self, z=None, T=None, ρ=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if ρ is None:
            ρ = self.ρ

        a_ni = self.a_ni
        b_ni = self.b_ni
        π = self.π
        k = self.k
        m = self.m
        η = self.f_ξ(z=z, T=T, ρ=ρ)[-1]
        m_bar = self.f_m_bar(z=z)
        ϵ_ij = self.ϵ_ij
        σ_ij = self.σ_ij

        a = a_ni[0] + (m_bar - 1) / m_bar * a_ni[1] + (m_bar - 1) / m_bar * (m_bar - 2) / m_bar * a_ni[2]
        b = b_ni[0] + (m_bar - 1) / m_bar * b_ni[1] + (m_bar - 1) / m_bar * (m_bar - 2) / m_bar * b_ni[2]

        I1 = [a[i] * η ** i for i in range(7)]
        I2 = [b[i] * η ** i for i in range(7)]

        I1 = np.sum(I1)
        I2 = np.sum(I2)

        Σ_i = 0
        for i in range(k):
            Σ_j = 0
            for j in range(k):
                Σ_j += z[i] * z[j] * m[i] * m[j] * (ϵ_ij[i][j] / T) * σ_ij[i][j] ** 3
            Σ_i += Σ_j
        Σ_1 = Σ_i

        Σ_i = 0
        for i in range(k):
            Σ_j = 0
            for j in range(k):
                Σ_j += z[i] * z[j] * m[i] * m[j] * (ϵ_ij[i][j] / T) ** 2 * σ_ij[i][j] ** 3
            Σ_i += Σ_j
        Σ_2 = Σ_i

        C1 = (1 + m_bar * (8 * η - 2 * η ** 2) / (1 - η) ** 4 + (1 - m_bar) * (
                20 * η - 27 * η ** 2 + 12 * η ** 3 - 2 * η ** 4) / ((1 - η) * (2 - η)) ** 2) ** -1

        return -2 * π * ρ * I1 * Σ_1 - π * ρ * m_bar * C1 * I2 * Σ_2

    def f_a_assoc(self, z=None, T=None, ρ=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if ρ is None:
            ρ = self.ρ

        k = self.k
        ϵ_AB_k = self.ϵ_AB_k
        Δ_AB_ij = self.f_Δ_AB_ij(z=z, T=T, ρ=ρ)

        def XA_find(XA_guess, n, Δ_AB_ij, ρ, z):
            # m = int(XA_guess.shape[1] / n)
            AB_matrix = np.asarray([[0., 1.],
                                    [1., 0.]])
            Σ_2 = np.zeros((n,), dtype='float_')
            XA = np.zeros_like(XA_guess)

            for i in range(n):
                Σ_2 = 0 * Σ_2
                for j in range(n):
                    Σ_2 += z[j] * (XA_guess[j, :] @ (Δ_AB_ij[i][j] * AB_matrix))
                XA[i, :] = 1 / (1 + ρ * Σ_2)

            return XA
        a_sites = 2  # 2B association?
        i_assoc = []
        for i in range(len(ϵ_AB_k)):
            if ϵ_AB_k[i] != 0:
                i_assoc.append(i)
        z_new = []
        for i in i_assoc:
            z_new.append(z[i])

        n_assoc = len(i_assoc)
        if n_assoc == 0 or n_assoc == 1:
            return 0

        XA = np.zeros((n_assoc, a_sites), dtype='float_')
        ctr = 0
        dif = 1000.
        XA_old = np.copy(XA)
        while (ctr < 500) and (dif > 1e-9):
            ctr += 1
            XA = XA_find(XA, n_assoc, Δ_AB_ij, ρ, z_new)
            dif = np.sum(abs(XA - XA_old))
            XA_old[:] = XA
        XA = XA.flatten()

        return sum([z[i] * sum([np.log(XA[j] - 1 / 2 * XA[j] + 1 / 2) for j in range(len(XA))]) for i in range(k)])

    def f_a_ion(self, z=None, T=None, ρ=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if ρ is None:
            ρ = self.ρ

        return 0

    def f_a_res(self, z=None, T=None, ρ=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if ρ is None:
            ρ = self.ρ

        a_hc = self.f_a_hc(z=z, T=T, ρ=ρ)
        a_disp = self.f_a_disp(z=z, T=T, ρ=ρ)
        a_assoc = self.f_a_assoc(z=z, T=T, ρ=ρ)
        a_ion = self.f_a_ion(z=z, T=T, ρ=ρ)

        return a_hc + a_disp + a_assoc + a_ion

    def f_da_dη(self, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η

        δ = .01
        h = η * δ
        a_res_1 = self.f_a_res(z=z, T=T, ρ=self.f_ρ(z=z, T=T, η=η - 2 * h))
        a_res_2 = self.f_a_res(z=z, T=T, ρ=self.f_ρ(z=z, T=T, η=η - 1 * h))
        a_res_3 = self.f_a_res(z=z, T=T, ρ=self.f_ρ(z=z, T=T, η=η + 1 * h))
        a_res_4 = self.f_a_res(z=z, T=T, ρ=self.f_ρ(z=z, T=T, η=η + 2 * h))

        return (a_res_1 - 8 * a_res_2 + 8 * a_res_3 - a_res_4) / (12 * h)

    def f_da_dT(self, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η

        δ = .01
        h = T * δ
        ρ = self.f_ρ(z=z, T=T, η=η)
        a_res_1 = self.f_a_res(z=z, T=T - 2 * h, ρ=ρ)
        a_res_2 = self.f_a_res(z=z, T=T - 1 * h, ρ=ρ)
        a_res_3 = self.f_a_res(z=z, T=T + 1 * h, ρ=ρ)
        a_res_4 = self.f_a_res(z=z, T=T + 2 * h, ρ=ρ)

        return (a_res_1 - 8 * a_res_2 + 8 * a_res_3 - a_res_4) / (12 * h)

    def f_da_dx(self, j, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η

        δ = .01
        h = z[j] * δ

        diff = [-2 * h, -h, h, 2 * h]
        z_new = np.zeros((4, 3))
        for n in range(4):
            for i in range(len(z)):
                if i == j:
                    z_new[n, i] = z[i] + diff[n]
                else:
                    z_new[n, i] = z[i]

        ρ = self.f_ρ(z=z, T=T, η=η)

        a_res_1 = self.f_a_res(z=z_new[0], T=T, ρ=ρ)
        a_res_2 = self.f_a_res(z=z_new[1], T=T, ρ=ρ)
        a_res_3 = self.f_a_res(z=z_new[2], T=T, ρ=ρ)
        a_res_4 = self.f_a_res(z=z_new[3], T=T, ρ=ρ)

        return (a_res_1 - 8 * a_res_2 + 8 * a_res_3 - a_res_4) / (12 * h)

    def f_Z(self, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η

        da_dη = self.f_da_dη(z=z, T=T, η=η)

        return 1 + η * da_dη

    def f_P(self, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η

        T = self.T
        Z = self.f_Z(z=z, T=T, η=η)
        ρ = self.f_ρ(z=z, T=T, η=η)
        kb = self.kb

        return Z * kb * T * ρ * 10 ** 30

    def f_find_η(self):

        def f(ηg):
            ηg = float(ηg)
            P = self.f_P(η=ηg)
            P_sys = self.P_sys
            return (P - P_sys) / 100000

        phase = self.phase
        if phase == 'liquid':
            ηg = .5
        elif phase == 'vapor':
            ηg = 10e-10
        else:
            print('Phase spelling probably wrong or phase is missing')
            ηg = .01
        res = root(f, np.array([ηg]))
        η = res.x[0]
        message = res.message

        if message != 'The solution converged.':
            print(message)

        self.η = η
        self.ρ = self.f_ρ()

    def f_h_res(self, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η

        Z = self.f_Z(z=z, T=T, η=η)
        da_dT = self.f_da_dT(z=z, T=T, η=η)

        return -T * da_dT + (Z - 1)

    def f_s_res(self, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η

        a_res = self.f_a_res(z=z, T=T, η=η)
        Z = self.f_Z(z=z, T=T, η=η)
        da_dT = self.f_da_dT(z=z, T=T, η=η)

        return -T * (da_dT + a_res / T) + np.log(Z)

    def f_g_res(self, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η
        ρ = self.f_ρ(z=z, T=T, η=η)

        a_res = self.f_a_res(z=z, T=T, ρ=ρ)
        Z = self.f_Z(z=z, T=T, η=η)
        return a_res + (Z - 1) - np.log(Z)

    def f_μ_res(self, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η

        ρ = self.f_ρ(z=z, T=T, η=η)
        a_res = self.f_a_res(z=z, T=T, ρ=ρ)
        Z = self.f_Z(z=z, T=T, η=η)
        da_dx = np.array([self.f_da_dx(i, z=z, T=T, η=η) for i in range(self.k)])

        Σ = np.sum(z * da_dx)

        μ_res = np.array([(a_res + (Z - 1) + da_dx[i] - Σ) * self.kb * T for i in range(len(z))])

        return μ_res

    def f_φ(self, z=None, T=None, η=None):

        if z is None:
            z = self.z
        if T is None:
            T = self.T
        if η is None:
            η = self.η

        T = self.T
        μ_res = self.f_μ_res(z=z, T=T, η=η)
        Z = self.f_Z(z=z, T=T, η=η)

        return np.exp(μ_res / self.kb / T - np.log(Z))


def flash_v3(x, y, T, P, prop_dic, flash_type='Bubble_T'):
    def eqs_to_solve(x, y, T, P):
        mix_l = PCSAFT_v3(T, x, prop_dic, phase='liquid', P_sys=P)
        mix_v = PCSAFT_v3(T, y, prop_dic, phase='vapor', P_sys=P)

        φ_l = mix_l.f_φ()
        φ_v = mix_v.f_φ()

        eqs = [(y[i] * φ_v[i] - x[i] * φ_l[i]) for i in range(len(y))]
        if flash_type[:-2] == 'Bubble':
            eqs.append(1 - np.sum([y[i] for i in range(len(y))]))
        elif flash_type[:-2] == 'Dew':
            eqs.append(1 - np.sum([x[i] for i in range(len(x))]))
        else:
            print('Wrong Flash Type')

        return eqs

    def f(w):
        if flash_type == 'Bubble_T':
            return eqs_to_solve(x, w[:-1], T, w[-1])
        elif flash_type == 'Bubble_P':
            return eqs_to_solve(x, w[:-1], w[-1], P)
        elif flash_type == 'Dew_T':
            return eqs_to_solve(w[:-1], y, T, w[-1])
        elif flash_type == 'Dew_P':
            return eqs_to_solve(w[:-1], y, w[-1], P)
        else:
            print('Wrong Flash Type')
            return None

    if flash_type == 'Bubble_T':
        guesses = list(y) + [P]
    elif flash_type == 'Bubble_P':
        guesses = list(y) + [T]
    elif flash_type == 'Dew_T':
        guesses = list(x) + [P]
    elif flash_type == 'Dew_P':
        guesses = list(x) + [T]
    else:
        print('Wrong Flash Type')
        guesses = []
    options = {'xtol': 1e-4, }
    ans = root(f, np.array([guesses]), options=options).x
    y_CO2, y_MEA, y_H2O = ans[:-1]
    P = ans[-1]
    P_CO2 = y_CO2*P

    return P_CO2



if __name__ == '__main__':
    x = [.1, .3, .6]
    T = 233.15
    yg = [.1, .3, .6]
    Pg = 1e5
    prop_dic = {
        'CO2': {'m_seg': 1, 'sigma': 3.7039, 'u_K': 150.03,
                'kappa_AB': 0,
                'eps_AB_k': 0},
        'MEA': {'m_seg': 1.6069, 'sigma': 3.5206, 'u_K': 191.42,
                'kappa_AB': 0,  # .037470,
                'eps_AB_k': 0,  # 2586.3,
                },
        'H2O': {'m_seg': 2.0020, 'sigma': 3.6184, 'u_K': 208.11,
                'kappa_AB': 0,  # .04509,
                'eps_AB_k': 0,  # 2425.67
                },
    }
    m = np.array([1, 1.6069, 2.0020])  # Number of segments
    σ = np.array([3.7039, 3.5206, 3.6184])  # Temperature-Independent segment diameter σ_i (Aᵒ)
    ϵ_k = np.array([150.03, 191.42, 208.11])  # Depth of pair potential / Boltzmann constant (K)
    k_ij = np.array([[0.00E+00, 3.00E-04, 1.15E-02],
                     [3.00E-04, 0.00E+00, 5.10E-03],
                     [1.15E-02, 5.10E-03, 0.00E+00]])
    print(flash_v3(x, yg, T, Pg, prop_dic, k_ij))
