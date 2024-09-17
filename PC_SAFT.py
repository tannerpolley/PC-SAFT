import numpy as np
import numdifftools as nd
from scipy.optimize import root, least_squares
from scipy.misc import derivative
from gekko import GEKKO
import pyomo.environ as pyo


class PCSAFT:
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
        self.η_diff = False
        self.T_diff = False
        self.x_diff = False
        self.d_static = σ * (1 - .12 * np.exp(-3 * ϵ_k / T))

        k = self.k

        if κ_AB is None:
            κ_AB = np.zeros(k)

        if ϵ_AB_k is None:
            ϵ_AB_k = np.zeros(k)

        # --------------------------------------- Intermediates --------------------------------------- #
        σ_ij = np.array([[1 / 2 * (σ[i] + σ[j]) for j in range(k)] for i in range(k)])
        self.σ_ij = σ_ij
        self.ϵ_ij = np.array([[(ϵ_k[i] * ϵ_k[j]) ** (1 / 2) * (1 - k_ij[i][j]) for j in range(k)] for i in range(k)])


        self.ϵ_AB_ij = np.array([[(ϵ_AB_k[i] + ϵ_AB_k[j]) / 2 for j in range(k)] for i in range(k)])
        self.κ_AB_ij = np.array([
            [
                np.sqrt(κ_AB[i] * κ_AB[j]) * (np.sqrt(σ_ij[i, i] * σ_ij[j, j]) / (1 / 2 * (σ_ij[i, i] + σ_ij[j, j]))) ** 3
                for j in range(k)]
            for i in range(k)])

        if P_sys is not None:
            self.P_sys = P_sys
            self.find_η()
        elif P_sys is None and η is not None:
            self.η = η
        elif P_sys is None and η is None:
            if phase == 'liquid':
                self.η = .4
            elif phase == 'vapor':
                self.η = .01
            print(
                'Warning, a default η had to be defined based on the given phase since so system pressure was given to iteratively find η')

    def m_bar(self):

        z = self.z
        m = self.m

        return sum(z * m)

    def d(self):
        T = self.T
        σ = self.σ
        ϵ_k = self.ϵ_k

        return σ * (1 - .12 * np.exp(-3 * ϵ_k / T))

    def d_og(self):
        T = self.T_og
        σ = self.σ
        ϵ_k = self.ϵ_k

        return σ * (1 - .12 * np.exp(-3 * ϵ_k / T))

    def ρ(self):

        z = self.z_og
        η = self.η
        d = self.d_static
        m = self.m
        k = self.k
        return 6 / self.π * η * (sum([z[i] * m[i] * d[i] ** 3 for i in range(k)])) ** (-1)

    def v(self):

        ρ = self.ρ()
        return self.N_A * 10 ** -30 / ρ

    def ξ(self):

        z = self.z
        d = self.d()
        ρ = self.ρ()
        m = self.m
        return np.array([self.π / 6 * ρ * np.sum([z[i] * m[i] * d[i] ** n for i in range(self.k)]) for n in range(4)])

    def g_hs_ij(self):

        d = self.d()
        ξ = self.ξ()

        return np.array([[(1 / (1 - ξ[3])) +
                          ((d[i] * d[j] / (d[i] + d[j])) * 3 * ξ[2] / (1 - ξ[3]) ** 2) +
                          ((d[i] * d[j] / (d[i] + d[j])) ** 2 * 2 * ξ[2] ** 2 / (1 - ξ[3]) ** 3)
                          for j in range(self.k)]
                         for i in range(self.k)])

    def d_ij(self):

        d = self.d()
        return np.array([[1 / 2 * (d[i] + d[j]) for j in range(self.k)] for i in range(self.k)])

    def Δ_AB_ij(self):
        T = self.T
        σ_ij = self.σ_ij
        g_hs_ij = self.g_hs_ij()
        κ_AB_ij = self.κ_AB_ij
        ϵ_AB_ij = self.ϵ_AB_ij
        k = self.k
        return np.array([[σ_ij[i][j] ** 3 * g_hs_ij[i][j] * κ_AB_ij[i][j] * (np.exp(ϵ_AB_ij[i][j] / T) - 1)
                          for j in range(k)] for i in range(k)])

    def a_hs(self):

        ξ = self.ξ()
        return 1 / ξ[0] * (3 * ξ[1] * ξ[2] / (1 - ξ[3]) + ξ[2] ** 3 / (ξ[3] * (1 - ξ[3]) ** 2) + (
                ξ[2] ** 3 / ξ[3] ** 2 - ξ[0]) * np.log(1 - ξ[3]))

    def a_hc(self):
        z = self.z

        k = self.k
        m = self.m
        m_bar = self.m_bar()
        g_hs_ij = self.g_hs_ij()
        a_hs = self.a_hs()

        return m_bar * a_hs - sum([z[i] * (m[i] - 1) * np.log(g_hs_ij[i][i]) for i in range(k)])

    def a_disp(self):

        T = self.T
        z = self.z
        η = self.ξ()[-1]

        a_ni = self.a_ni
        b_ni = self.b_ni
        π = self.π
        k = self.k
        ρ = self.ρ()
        m = self.m
        m̄ = self.m_bar()
        ϵ_ij = self.ϵ_ij
        σ_ij = self.σ_ij

        a = a_ni[0] + (m̄ - 1) / m̄ * a_ni[1] + (m̄ - 1) / m̄ * (m̄ - 2) / m̄ * a_ni[2]
        b = b_ni[0] + (m̄ - 1) / m̄ * b_ni[1] + (m̄ - 1) / m̄ * (m̄ - 2) / m̄ * b_ni[2]

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

        C1 = (1 + m̄ * (8 * η - 2 * η ** 2) / (1 - η) ** 4 + (1 - m̄) * (
                20 * η - 27 * η ** 2 + 12 * η ** 3 - 2 * η ** 4) / ((1 - η) * (2 - η)) ** 2) ** -1

        return -2 * π * ρ * I1 * Σ_1 - π * ρ * m̄ * C1 * I2 * Σ_2

    def a_assoc(self):

        z = self.z
        k = self.k
        ρ = self.ρ()
        ϵ_AB_k = self.ϵ_AB_k
        Δ_AB_ij = self.Δ_AB_ij()

        def XA_find(XA_guess, Δ_AB_ij, ρ, z):
            n = len(XA_guess)
            AB_matrix = np.asarray([[0., 0., 0.],
                                    [0., 0., 1.],
                                    [0., 1., 0.]])
            XA = np.zeros((n, n))
            for i in range(n):
                XA[i, :] = XA_guess[i]
            eqs = []
            for i in range(n):
                Σ_2 = 0
                for j in range(n):
                    Σ_2 += z[j] * (XA[j, :] @ (Δ_AB_ij[i][j] * AB_matrix))
                if XA[i, i] == 0.:
                    eqs.append(XA[i, i] - 0.)
                else:
                    eqs.append(XA[i, i] - (1 / (1 + ρ * Σ_2[-1])))

            return eqs

        a_sites = 2  # 2B association?
        i_assoc = []
        for i in range(len(ϵ_AB_k)):
            if ϵ_AB_k[i] != 0:
                i_assoc.append(i)

        n_assoc = len(i_assoc)
        if n_assoc == 0 or n_assoc == 1:
            return 0

        XA_guess = np.zeros(k, dtype='float_')
        for i in range(k):
            if Δ_AB_ij[i, i] == 0.:
                XA_guess[i] = 0.
            else:
                XA_guess[i] = (-1 + np.sqrt(1+8*ρ*Δ_AB_ij[i, i]))/(4*ρ*Δ_AB_ij[i, i])

        XA_i = root(XA_find, XA_guess, args=(Δ_AB_ij, ρ, z)).x
        XA = np.zeros((k, k))
        for i in range(k):
            XA[i, :] = XA_i
        XA = XA.T
        XA_final = []
        for i in range(k):
            for j in range(n_assoc):
                if XA[i, j] != 0.:
                    XA_final.append(XA[i, j])
        XA = XA_final
        return sum([z[i_assoc[i]] * sum([np.log(XA[i*a_sites+j]) - 1 / 2 * XA[i*a_sites+j] for j in range(a_sites)]) + 1 / 2 for i in range(n_assoc)])

    def a_ion(self):
        return 0

    def a_res(self):

        a_hc = self.a_hc()
        a_disp = self.a_disp()
        a_assoc = self.a_assoc()
        a_ion = self.a_ion()

        return a_hc + a_disp + a_assoc + a_ion

    def da_dη(self):

        η = self.η

        self.η_og = self.η

        def f(η_diff):
            self.η = η_diff
            return self.a_res()

        self.η = self.η_og

        return derivative(f, η, dx=1e-3)

    def da_dx(self):

        z = self.z
        self.z_og = self.z
        da_dx = []
        for k in range(len(z)):

            def f(zk):

                z_new = []
                for i in range(len(z)):
                    if i == k:
                        z_new.append(zk)
                    else:
                        z_new.append(z[i])

                self.z = z_new

                return self.a_res()

            self.z = self.z_og

            da_dx.append(derivative(f, z[k], dx=1e-3))

        self.z = self.z_og

        return np.array(da_dx)

    def da_dT(self):

        T = self.T
        self.T_og = self.T

        def f(T_diff):
            self.T = T_diff
            return self.a_res()

        self.T = self.T_og
        return derivative(f, T, dx=1e-3)

    def Z(self):
        η = self.η

        da_dη = self.da_dη()

        self.η = self.η_og

        return 1 + η * da_dη

    def P(self):

        T = self.T
        Z = self.Z()
        ρ = self.ρ()
        kb = self.kb

        return Z * kb * T * ρ * 10 ** 30

    def find_η(self):

        def f(ηg):
            self.η = float(ηg)
            P = self.P()
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
        η = root(f, np.array([ηg])).x[0]

        self.η = η

    def h_res(self):

        T = self.T
        self.T_og = self.T

        Z = self.Z()
        da_dT = self.da_dT()

        self.T = self.T_og

        return -T * da_dT + (Z - 1)

    def s_res(self):

        T = self.T
        self.T_og = self.T

        a_res = self.a_res()
        Z = self.Z()
        da_dT = self.da_dT()

        self.T = self.T_og

        return -T * (da_dT + a_res / T) + np.log(Z)

    def g_res(self):
        a_res = self.a_res()
        Z = self.Z()
        return a_res + (Z - 1) - np.log(Z)

    def μ_res(self):

        z = self.z
        T = self.T

        a_res = self.a_res()
        Z = self.Z()
        da_dx = self.da_dx()

        Σ = np.sum([z[j] * da_dx[j] for j in range(len(z))])

        μ_res = [(a_res + (Z - 1) + da_dx[i] - Σ) * self.kb * T for i in range(len(z))]

        return np.array(μ_res)

    def φ(self):

        T = self.T
        μ_res = self.μ_res()
        Z = self.Z()

        return np.exp(μ_res / self.kb / T - np.log(Z))


def flash(x, y, T, P, prop_dic, flash_type='Bubble_P'):
    def eqs_to_solve(x, y, T, P):
        # print(κ_AB)
        mix_l = PCSAFT(T, x, prop_dic, phase='liquid', P_sys=P)
        mix_v = PCSAFT(T, y, prop_dic, phase='vapor', P_sys=P)

        φ_l = mix_l.φ()
        φ_v = mix_v.φ()
        # print(mix_l.a_assoc())
        # print(mix_l.a_res(), mix_l.a_hc(), mix_l.a_disp(), mix_l.a_assoc(), mix_l.a_ion())
        # print(mix_l.a_res(), mix_l.a_hc() + mix_l.a_disp(), mix_l.a_hc() + mix_l.a_disp() + mix_l.a_assoc())

        eqs = [(y[i] * φ_v[i] - x[i] * φ_l[i]) for i in range(len(y))]
        if flash_type[:-2] == 'Bubble':
            eqs.append(1 - np.sum([y[i] for i in range(len(y))]))
        elif flash_type[:-2] == 'Dew':
            eqs.append(1 - np.sum([x[i] for i in range(len(x))]))
        else:
            print('Wrong Flash Type')

        return eqs

    def f(w):
        if flash_type == 'Bubble_P':
            return eqs_to_solve(x, w[:-1], T, w[-1])
        elif flash_type == 'Bubble_T':
            return eqs_to_solve(x, w[:-1], w[-1], P)
        elif flash_type == 'Dew_P':
            return eqs_to_solve(w[:-1], y, T, w[-1])
        elif flash_type == 'Dew_T':
            return eqs_to_solve(w[:-1], y, w[-1], P)
        else:
            print('Wrong Flash Type')
            return None

    if flash_type == 'Bubble_P':
        guesses = list(y) + [P]
    elif flash_type == 'Bubble_T':
        guesses = list(y) + [T]
    elif flash_type == 'Dew_P':
        guesses = list(x) + [P]
    elif flash_type == 'Dew_T':
        guesses = list(x) + [T]
    else:
        print('Wrong Flash Type')
        guesses = []
    options = {'xtol': 1e-4, }
    result = root(f, np.array([guesses]), options=options)

    ans, success = result.x, result.success

    if not success:
        print('failed')
        return np.nan, [np.nan, np.nan, np.nan]

    # mix_l = PCSAFT(T, x, m, σ, ϵ_k, k_ij, phase='liquid', P_sys=P, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)
    # mix_v = PCSAFT(T, y, m, σ, ϵ_k, k_ij, phase='vapor', P_sys=P, κ_AB=κ_AB, ϵ_AB_k=ϵ_AB_k)

    return ans[-1], ans[:-1]


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
    print(flash(x, yg, T, Pg, prop_dic, k_ij))
